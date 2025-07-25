from copy import deepcopy
import numpy as np
import yaml
import builtins

# original_print = builtins.print

# def custom_print(*args, **kwargs):
#     # You can add conditions here to filter specific prints
#     if not any(isinstance(arg, torch.Size) for arg in args):
#         original_print(*args, **kwargs)

# builtins.print = custom_print
import argparse
import math
import torch
from collections import defaultdict
import torch.nn as nn
from lib import loaders
from tqdm.auto import tqdm
from ema_pytorch import EMA
from numpy import *
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torch_pruning as tp
import torchvision as tv

# from test_pruning import pruning

from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

torch.backends.cuda.matmul.allow_tf32 = True
## The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def variance_of_laplacian(img):
    # img: (B, 1, H, W)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
    lap = F.conv2d(img, laplacian_kernel, padding=1)
    var = torch.var(lap, dim=[1,2,3])  # per-image variance
    return var

def calc_relative_loss(pred, target, all_relative, sample_size=2000):
    # print("Target min:", target.min().item())
    # print("Target max:", target.max().item())
    relative_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
    relative_error = torch.sqrt(relative_error)

    # print("Minimum relative error:", relative_error.min().item())

    clipped_error = torch.clamp(relative_error, min=0.0, max=1)
    
    # scaled_error = torch.tanh(relative_error)
    relative_error_np = clipped_error.detach().cpu().numpy().flatten()
    sampled_data = np.random.choice(relative_error_np, size=sample_size, replace=False)
    
    all_relative.append(sampled_data)

def calc_loss_test(pred1, pred2, target, metrics, error="MSE"):
    pred3 = pred1
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()

    loss1 = criterion(pred1, target)/criterion(target, 0*target)
    loss2 = torch.sqrt(criterion(pred2, target))
    # loss_mae = mae_loss(pred3, target)

    # vol1 = variance_of_laplacian(pred1)
    # vol_target = variance_of_laplacian(target)

    # vol = vol1/vol_target
    # vol_value = vol.mean().item()
    # print(vol.data.cpu().numpy() * target.size(0))

    ssim1 = ssim(pred1, target)
    # print(ssim1)
    #ssim2 = ssim(pred2, target)

    psnr1 = psnr(pred1,target)

    metrics['nmse'] += loss1.data.cpu().numpy() * target.size(0)
    metrics['rmse'] += loss2.data.cpu().numpy() * target.size(0)
    # metrics['mae']  += loss_mae.data.cpu().numpy() * target.size(0)
    metrics['ssim'] += ssim1.data.cpu().numpy() * target.size(0)
    metrics['psnr'] += psnr1.data.cpu().numpy() * target.size(0)
    # metrics['vol'] += vol_value * target.size(0)

    return [loss1,loss2]

def print_metrics_test(metrics, epoch_samples, error):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format("Test"+" "+error, ", ".join(outputs)))

def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def brightest_point_distance(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """
    计算两个形状为 [1, 1, H, W] 的图像中最亮点的欧几里得距离。

    参数:
        gt: torch.Tensor，ground truth 图像 [1, 1, H, W]
        pred: torch.Tensor，预测图像 [1, 1, H, W]

    返回:
        float: 最亮点之间的欧几里得距离
    """
    gt_2d = gt.squeeze()    # [256, 256]
    pred_2d = pred.squeeze()

    gt_idx = torch.argmax(gt_2d)
    pred_idx = torch.argmax(pred_2d)

    gt_y, gt_x = divmod(gt_idx.item(), gt_2d.shape[1])
    pred_y, pred_x = divmod(pred_idx.item(), pred_2d.shape[1])

    distance = math.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    return distance


# Colors for all 20 parts
part_colors = [[0, 0, 0], [255, 85, 0],  [255, 170, 0],
               [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0],
               [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255],
               [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170],
               [255, 0, 255], [255, 85, 255], [255, 170, 255],
               [0, 255, 255], [85, 255, 255], [170, 255, 255]]

def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )
    
    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    carsDPM=unet_cfg.DPMCARK
                    )
        
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=model_cfg.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        # ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    # ldm.init_from_ckpt(cfg.sampler.ckpt_path, use_ema=cfg.sampler.get('use_ema', True))

    data_cfg = cfg.data

    if data_cfg['name'] == 'edge':
        dataset = EdgeDatasetTest(
            data_root=data_cfg.img_folder,
            image_size=model_cfg.image_size,
        )
        # dataset = torch.utils.data.ConcatDataset([dataset] * 5)
    elif data_cfg['name'] == 'radio':
        dataset = loaders.RadioUNet_c(phase="test", dir_dataset="/home/disk01/qmzhang/RadioMapSeer/")

    elif data_cfg['name'] == 'IRT4':
        dataset = loaders.RadioUNet_c_sprseIRT4(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="IRT4",carsSimul="no",carsInput="no")
    elif data_cfg['name'] == 'IRT4K':
        dataset = loaders.RadioUNet_c_sprseIRT4_K2(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="IRT4",carsSimul="no",carsInput="K2")
    elif data_cfg['name'] == 'DPMK':
        dataset = loaders.RadioUNet_c_K2(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="DPM",carsSimul="no",carsInput="K2")
    elif data_cfg['name'] == 'DPMCAR': #参数默认进入car_gain_image
        dataset = loaders.RadioUNet_c_WithCar_NOK_or_K(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="DPM", have_K2="no")
    elif data_cfg['name'] == 'DPMCARK': #参数默认进入car_gain_image
        dataset = loaders.RadioUNet_c_WithCar_NOK_or_K(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="DPM", have_K2="yes")
    elif data_cfg['name'] == 'MASK':
        dataset = loaders.RadioUNet_s(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/",mask=True)
    elif data_cfg['name'] == 'MASK_R':
        dataset = loaders.RadioUNet_s(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/")
    elif data_cfg['name'] == 'RANDOM':
        dataset = loaders.RadioUNet_s_random(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", mask=True)
    elif data_cfg['name'] == 'VERTEX':
        dataset = loaders.RadioUNet_s_vertex(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", mask=True)
    elif data_cfg['name'] == 'VERTEX_R':
        dataset = loaders.RadioUNet_s_vertex(phase="test",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/")
    else:
        raise NotImplementedError
    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))


    sampler_cfg = cfg.sampler
    sampler = Sampler(
        ldm, dl, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder,cfg=cfg,
    )
    sampler.sample()
    
    # BY PLZHENG
    # useless code
    # if data_cfg.name == 'cityscapes' or data_cfg.name == 'sr' or data_cfg.name == 'edge':
    #     exit()
    # else:
    #     assert len(os.listdir(sampler_cfg.target_path)) > 0, "{} have no image !".format(sampler_cfg.target_path)
    #     sampler.cal_fid(target_path=sampler_cfg.target_path)
    # pass


def nmse(res, target):
    criterion = nn.MSELoss()
    return criterion(res, target) / criterion(target, 0 * target)


class Sampler(object):
    def __init__(
            self,
            model,
            data_loader,
            sample_num=1000,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            # BY PLZHENG
            # use fp16
            mixed_precision= 'fp16' if cfg.sampler.use_fp16 else 'no',
            
            kwargs_handlers=[ddp_handler],
        )
        # BY PLZHENG
        print(f"***using fp16 while sampling: [{cfg.sampler.use_fp16}]***")
        
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45

        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)

        self.image_size = model.image_size
        self.cfg = cfg

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.results_folder = Path(results_folder)
        # self.results_folder_cond = Path(results_folder+'_cond')
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)
            # self.results_folder_cond.mkdir(exist_ok=True, parents=True)
        # Load model and checkpoint
        data = torch.load(cfg.sampler.ckpt_path, map_location=lambda storage, loc: storage)
        model = self.model

        # Load state dict
        if cfg.sampler.use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            model.load_state_dict(data['model'], strict=False)
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

        # Calculate initial parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.accelerator.is_main_process:
            print(f'Initial Total Parameters: {total_params:,}')
            print(f'Initial Trainable Parameters: {trainable_params:,}')

        # Prune model
        #model.model = pruning(model.model.cpu(), 0.9).cuda()
            
        
        # 先量化 再accelerator封装
        # print('Begin QAT...')
        
        # model.model = torch.quantization.quantize_dynamic(
        #     self.model.model,  # 需要量化的子模型
        #     {torch.nn.Conv2d,torch.nn.BatchNorm2d},  # 指定量化模块，这里只指定Linear层
        #     dtype=torch.quint4x2  # 使用int8量化
        # )   
        # print(self.model.model)

        self.model = self.accelerator.prepare(model)
        
        # Calculate pruned parameters
        # pruned_model = self.accelerator.unwrap_model(self.model)
        # total_params = sum(p.numel() for p in pruned_model.parameters()) 
        # trainable_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)

        # if self.accelerator.is_main_process:
        #     print(f'Pruned Total Parameters: {total_params:,}')
        #     print(f'Pruned Trainable Parameters: {trainable_params:,}')

    def sample(self):
        metrics = defaultdict(float)
        accelerator = self.accelerator
        device = accelerator.device
        epoch_samples = 0
        batch_num = self.batch_num
        self.model.eval()
        all_relative_errors = []
        with torch.no_grad():
            with accelerator.autocast():
                psnr = 0.
                num = 0
                nmse_ = []
                
                accelerator.print("\n-------------------------------------\n")
                # BY PLZHENG
                # ['WARM_UP', 'INFERENCE']
                Stage = ['WARM_UP', 'INFERENCE']
                for stage in Stage:
                    # clone the test_loader
                    tmp_dl = deepcopy(self.dl)
                    # prepare for 'WARM_UP'
                    if stage == 'WARM_UP':
                        if self.cfg.sampler.warm_up_steps == 0:
                            accelerator.print("***no warm up!***")
                            continue
                        accelerator.print("***starting warm up the device...***")
                        warm_up_stop_idx = builtins.max(0, self.cfg.sampler.warm_up_steps // self.cfg.model.sampling_timesteps)
                        
                    # prepare for 'INFERENCE'
                    elif stage == 'INFERENCE':
                        accelerator.print("***starting inference stage...***")
                        accelerator.print(f"***dataloader length : {len(self.dl)}***")
                        inference_stop_idx = builtins.min(self.cfg.sampler.inference_stop_idx, len(self.dl) - 1)
                        whole_sample_times = []
                        # use timer
                        if self.cfg.sampler.use_timer:
                            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            times = torch.zeros(inference_stop_idx + 1, device=device)
                    
                    total_dist = []
                    # running the model
                    for idx, batch in tqdm(enumerate(tmp_dl),  disable=not self.accelerator.is_main_process):

                        for key in batch.keys():
                            if isinstance(batch[key], torch.Tensor):
                                batch[key].to(device)


                        # image = batch["image"]
                        # image = unnormalize_to_zero_to_one(image)
                        cond = batch['cond']    
                        GT = batch['image']     
                        #print(GT.size())
                        # print(batch["raw_size"])
                        # raw_w = batch["raw_size"][0].item()      # default batch size = 1
                        # raw_h = batch["raw_size"][1].item()
                        img_name = batch["img_name"][0]

                        mask = batch['ori_mask'] if 'ori_mask' in batch else None
                        bs = cond.shape[0]
                        
                        # BY PLZHENG 
                        # start the timer
                        if stage == 'INFERENCE' and self.cfg.sampler.use_timer:
                            starter.record()
                        
                        # INFERENCE
                        if self.cfg.sampler.sample_type == 'whole':
                            batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
                        elif self.cfg.sampler.sample_type == 'slide':
                            start_time = time.time()
                            batch_pred = self.slide_sample(cond, crop_size=self.cfg.sampler.get('crop_size', [256, 256]), stride=self.cfg.sampler.stride, mask=mask)
                            end_time = time.time()
                            whole_sample_times.append(end_time - start_time)
                        else:
                            raise NotImplementedError
                        
                        # BY PLZHENG
                        # stop the timer

                        # print(GT.shape)     #torch.Size([1, 1, 256, 256])
                        # print(batch_pred.shape)     #torch.Size([1, 1, 256, 256])
                        

                        dist = brightest_point_distance(GT, batch_pred)
                        total_dist.append(dist)

                        if stage == 'INFERENCE' and self.cfg.sampler.use_timer:
                            ender.record()
                            torch.cuda.synchronize()
                            curr_time = starter.elapsed_time(ender) # 计算时间
                            times[idx] = curr_time
                        
                            
                        # BY PLZHENG
                        # process 'WARM_UP'
                        if stage == 'WARM_UP' and idx == warm_up_stop_idx:
                            accelerator.print("***device is ready!***")
                            break
                        
                        
                        # BY PLZHENG
                        # process 'INFERENCE'
                        if stage == 'INFERENCE':
                            calc_loss_test(batch_pred.cpu(), batch_pred.cpu(), (GT * 0.5 + 0.5).cpu(), metrics,
                                        'mse')
                            
                            # calc_loss_test(batch_pred.cpu(), batch_pred.cpu(), GT.cpu(), metrics,
                            #             'mse')

                            
                            calc_relative_loss(batch_pred.cpu(), (GT * 0.5 + 0.5).cpu(), all_relative_errors)

                            epoch_samples += batch_pred.size(0)
                            # print(epoch_samples)
                            for j, (img, c) in enumerate(zip(batch_pred, cond)):
                                file_name = self.results_folder / img_name
                                """
                                ==========================================================================================
                                """
                                # img = (img + 1) / 2
                                tv.utils.save_image(img, str(file_name)[:-4] + ".png")
                                nmse_.append(nmse(img.cpu(), (GT[0]*0.5+0.5).cpu()))
                            if idx == inference_stop_idx:
                                import numpy as np
                                import matplotlib.pyplot as plt
                                import seaborn as sns
                                
                                # all_errors = np.concatenate(all_relative_errors)
                                # filtered_errors = all_errors[all_errors > 1e-3]
                                # plt.hist(filtered_errors, bins=100, color='blue', edgecolor='black', alpha=0.4, density=True)
                                # sns.kdeplot(filtered_errors, color='red', lw=1.5)
                                # # plt.xlim(0, 1) 
                                # # sns.kdeplot(all_errors, bw_adjust=0.5, fill=True)
                                # plt.xlabel("Relative Error")
                                # plt.ylabel("Density")
                                # plt.title("Relative Error Distribution")
                                # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                                # plt.savefig("relative_error_distribution_vertical.png", dpi=300, bbox_inches='tight')    
                                # plt.show()


                                if self.cfg.sampler.use_timer:
                                    # accelerator.print("\n-------------------------------------\n")
                                    # accelerator.print("times : ", times)
                                    # mean_time = times.mean().item()
                                    # accelerator.print("\n-------------------------------------\n")
                                    # accelerator.print("***Inference time: {:.6f} ms***".format(mean_time))
                                    
                                    times_all = accelerator.gather(times)
                                    if accelerator.is_main_process:
                                        mean_time = times_all.mean().item()
                                        accelerator.print("\n-------------------------------------\n")
                                        accelerator.print("***Global Inference time: {:.6f} ms***".format(mean_time))
                                break  #by zhangqiming
                        
                    if self.cfg.sampler.sample_type == 'slide':
                        avg_time = sum(whole_sample_times)/ len(whole_sample_times)
                        print(f'Average whole sample time: {avg_time:.4f} seconds')
        print_metrics_test(metrics, epoch_samples, 'mse')
        avg_dist = sum(total_dist) / len(total_dist)
        print(f"平均最亮点距离为: {avg_dist}")
        accelerator.print('sampling complete')
        # accelerator.print(f'nmse_: {mean(nmse_)}')

    # ----------------------------------waiting revision------------------------------------
    def slide_sample(self, inputs, crop_size, stride, mask=None):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        # aux_out2 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                # print(y1, h_crop, h_img)
                y2 = builtins.min(y1 + h_crop, h_img)
                
                x2 = builtins.min(x1 + w_crop, w_img)
                y1 = builtins.max(y2 - h_crop, 0)
                x1 = builtins.max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img, mask=mask)
                    e1 = e2 = None
                    aux_out = None
                elif isinstance(self.model, nn.Module):
                    crop_seg_logit = self.model.sample(batch_size=1, cond=crop_img, mask=mask)
                    e1 = e2 = None
                    aux_out = None
                else:
                    raise NotImplementedError
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                if aux_out is not None:
                    aux_out1 += F.pad(aux_out,
                                   (int(x1), int(aux_out1.shape[3] - x2), int(y1),
                                    int(aux_out1.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        # torch.save(count_mat, '/home/yyf/Workspace/edge_detection/codes/Mask-Conditioned-Latent-Space-Diffusion/checkpoints/count.pt')
        seg_logits = preds / count_mat
        aux_out1 = aux_out1 / count_mat
        # aux_out2 = aux_out2 / count_mat
        if aux_out is not None:
            return seg_logits, aux_out1
        return seg_logits

    def whole_sample(self, inputs, raw_size, mask=None):

        inputs = F.interpolate(inputs, size=(416, 416), mode='bilinear', align_corners=True)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        seg_logits = F.interpolate(seg_logits, size=raw_size, mode='bilinear', align_corners=True)
        return seg_logits


    def cal_fid(self, target_path):
        command = 'fidelity -g 0 -f -i -b {} --input1 {} --input2 {}'\
            .format(self.batch_size, str(self.results_folder), target_path)
        os.system(command)

    def rk45_sample(self, batch_size):
        with torch.no_grad():
            # Initial sample
            # z = torch.randn(batch_size, 3, *(self.image_size))
            shape = (batch_size, 3, *(self.image_size))
            ode_sampler = get_ode_sampler(method='RK45')
            x, nfe = ode_sampler(model=self.model, shape=shape)
            x = unnormalize_to_zero_to_one(x)
            x.clamp_(0., 1.)
            return x, nfe

def get_ode_sampler(rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t, model_type='const'):
    """Get the drift function of the reverse-time SDE."""
    # score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # rsde = sde.reverse(score_fn, probability_flow=True)
    pred = model(x, t)
    if model_type == 'const':
        drift = pred
    elif model_type == 'linear':
        K, C = pred.chunk(2, dim=1)
        drift = K * t + C
    return drift

  def ode_sampler(model, shape):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = torch.randn(*shape)
      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        # vec_t = torch.ones(shape[0], device=x.device) * t
        vec_t = torch.ones(shape[0], device=x.device) * t * 1000
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (1, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      # if denoise:
      #   x = denoise_update_fn(model, x)

      # x = inverse_scaler(x)
      return x, nfe

  return ode_sampler

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass