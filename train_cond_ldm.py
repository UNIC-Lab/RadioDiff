import yaml
import argparse
import math
import torch
from lib import loaders
import torch.nn as nn
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.ema import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode

# from accelerate import AccelerateConfig
import os

import os
import yaml


acconfig = {
    'compute_environment': 'LOCAL_MACHINE',
    'debug': False,
    'distributed_type': 'NO',
    'downcast_bf16': 'no',
    'enable_cpu_affinity': True,
    'gpu_ids': '3',  #直接改这个
    'machine_rank': 0,
    'main_training_function': 'main',
    'mixed_precision': 'no',
    'num_machines': 1,
    'num_processes': 1,
    'rdzv_backend': 'static',
    'same_network': True,
    'tpu_env': [],
    'tpu_use_cluster': False,
    'tpu_use_sudo': False,
    'use_cpu': False,
}

# 设置配置文件路径
config_dir = os.path.expanduser('/home/zhangqiming/.cache/huggingface/accelerate')
os.makedirs(config_dir, exist_ok=True)

config_path = os.path.join(config_dir, 'default_config.yaml')

# 写入 YAML 文件
with open(config_path, 'w') as f:
    yaml.dump(acconfig, f)

print(f"配置写入成功：{config_path}")




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

def save_model_freeze_report(model, file_path='model_freeze_report.txt'):
    with open(file_path, 'w') as f:
        total_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if not param.requires_grad:
                frozen_params += num_params
            status = 'Frozen' if not param.requires_grad else 'Trainable'
            f.write(f'{name:60} | {status:8} | Params: {num_params}\n')

        f.write('\n')
        f.write(f'Total parameters: {total_params}\n')
        f.write(f'Frozen parameters: {frozen_params}\n')
        f.write(f'Trainable parameters: {total_params - frozen_params}\n')
        f.write(f'Frozen ratio: {frozen_params / total_params:.2%}\n')

    print(f'Model freeze report saved to {file_path}')


def main(args):
    cfg = CfgNode(args.cfg)
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
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
                    carsDPM=unet_cfg.DPMCARK,
                    cfg=unet_cfg,
                    )



    elif model_cfg.model_name == 'cond_unet_8bit':
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
                    )
        # Convert model to 8-bit quantization
        # unet = torch.quantization.quantize_dynamic(
        #     unet, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        # )
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
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    data_cfg = cfg.data

    if data_cfg['name'] == 'edge':
        dataset = EdgeDataset(
            data_root=data_cfg.img_folder,
            image_size=model_cfg.image_size,
            augment_horizontal_flip=data_cfg.augment_horizontal_flip,
            cfg=data_cfg
        )
    #=======================这个是用来之前做K图关键数据集加载部分 四月份重要部分====================================#
    elif data_cfg['name'] == 'radio':
        dataset = loaders.RadioUNet_c(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/",simulation="DPM",carsSimul="no",carsInput="no")
    elif data_cfg['name'] == 'IRT4':
        dataset = loaders.RadioUNet_c_sprseIRT4(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="IRT4",carsSimul="no",carsInput="no")
    # elif data_cfg['name'] == 'CARIRT4':
    #     dataset = loaders.RadioUNet_c_sprseIRT4(phase="train",dir_dataset="/home/DataDisk/qmzhang/RadioMapSeer/", simulation="IRT4",carsSimul="no",carsInput="no")
    elif data_cfg['name'] == 'IRT4K':
        dataset = loaders.RadioUNet_c_sprseIRT4_K2(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="IRT4",carsSimul="no",carsInput="K2")
    elif data_cfg['name'] == 'DPMK':
        dataset = loaders.RadioUNet_c_K2(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="DPM",carsSimul="no",carsInput="K2")
    elif data_cfg['name'] == 'DPMCAR':
        dataset = loaders.RadioUNet_c_WithCar_NOK_or_K(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="DPM", have_K2="no")
    elif data_cfg['name'] == 'DPMCARK':
        dataset = loaders.RadioUNet_c_WithCar_NOK_or_K(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", simulation="DPM", have_K2="yes")

    #=======================这一部分是用来加载建筑物边缘采样的关键数据集部分 五月份重要部分=============================#
    elif data_cfg['name'] == 'MASK':
        dataset = loaders.RadioUNet_s(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/",mask=True)
    elif data_cfg['name'] == 'MASK_R':
        dataset = loaders.RadioUNet_s(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/")
    elif data_cfg['name'] == 'RANDOM':
        dataset = loaders.RadioUNet_s_random(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", mask=True)
    elif data_cfg['name'] == 'VERTEX':
        dataset = loaders.RadioUNet_s_vertex(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/", mask=True)
    elif data_cfg['name'] == 'VERTEX_R':
        dataset = loaders.RadioUNet_s_vertex(phase="train",dir_dataset="/home/disk01/qmzhang/RadioMapSeer/")
    else:
        raise NotImplementedError
    
    # dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True, pin_memory=True,
    #                 num_workers=data_cfg.get('num_workers', 2))
    
    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True)

    # # by zhangqiming
    # batch = next(iter(dl))
    # images = batch['image']  # 提取图像
    # conds = batch['cond']  # 提取条件信息
    # names = batch['img_name']  # 提取图片名称

    # print(f"======={conds.shape}======")
    # import torch
    # import os
    # from torchvision.utils import save_image
    # save_dir = './saved_channels/'
    # os.makedirs(save_dir, exist_ok=True)
    # for i in range(conds.size(0)):  # 遍历每张图像
    #     for j in range(conds.size(1)):  # 遍历每个通道
    #         # 获取当前通道的图像
    #         channel_image = conds[i, j, :, :]  # shape: [256, 256]
    #         # 构造保存的文件名
    #         filename = f"{save_dir}/image_{i}_channel_{j}.png"
    #         # 保存图像
    #         save_image(channel_image, filename)
    #         print(f"Saved {filename}")


    
    train_cfg = cfg.trainer
    # from test_pruning import pruning
    trainer = Trainer(
        ldm, dl, train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg.lr, train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every, results_folder=train_cfg.results_folder,
        amp=train_cfg.amp, fp16=train_cfg.fp16, log_freq=train_cfg.log_freq, cfg=cfg,
        resume_milestone=train_cfg.resume_milestone,
        train_wd=train_cfg.get('weight_decay', 1e-4)
    )
    if train_cfg.test_before:
        if trainer.accelerator.is_main_process:
            with torch.no_grad():
                for datatmp in dl:
                    break
                print(type(datatmp))
                if isinstance(trainer.model, nn.parallel.DistributedDataParallel):
                    all_images, *_ = trainer.model.module.sample(batch_size=datatmp['cond'].shape[0],
                                                  cond=datatmp['cond'].to(trainer.accelerator.device),
                                                  mask=datatmp['ori_mask'].to(trainer.accelerator.device) if 'ori_mask' in datatmp else None)
                elif isinstance(trainer.model, nn.Module):
                    all_images, *_ = trainer.model.sample(batch_size=datatmp['cond'].shape[0],
                                                  cond=datatmp['cond'].to(trainer.accelerator.device),
                                                  mask=datatmp['ori_mask'].to(trainer.accelerator.device) if 'ori_mask' in datatmp else None)

            # all_images = torch.cat(all_images_list, dim = 0)
            nrow = 2 ** math.floor(math.log2(math.sqrt(data_cfg.batch_size)))
            tv.utils.save_image(all_images, str(trainer.results_folder / f'sample-{train_cfg.resume_milestone}_{model_cfg.sampling_timesteps}.png'), nrow=nrow)
            torch.cuda.empty_cache()
    trainer.train()
    pass


class Trainer(object):
    def __init__(
            self,
            model,
            data_loader,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_wd=1e-4,
            train_num_steps=100000,
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_freq=20,
            resume_milestone=0,
            cfg={},
    ):
        
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            kwargs_handlers=[ddp_handler],
        )
        self.enable_resume = cfg.trainer.get('enable_resume', False)
        self.accelerator.native_amp = amp

        self.model = model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        self.image_size = model.image_size

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        #by zhangqiming  finetune入口
        if cfg.finetune.ckpt_path is not None:
            data = torch.load(cfg.finetune.ckpt_path, map_location=lambda storage, loc: storage)
            model = self.model
            model.load_state_dict(data['model'], strict=False)

            if 'scale_factor' in data['model']:
                model.scale_factor = data['model']['scale_factor']

        # optimizer
        self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=train_lr, weight_decay=train_wd)
        
        # save_model_freeze_report(model)

        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.96, cfg.trainer.min_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.ema = EMA(model, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # self.model, self.opt, self.lr_scheduler = \
        #     self.accelerator.prepare(self.model, self.opt, self.lr_scheduler)

        self.model, self.opt, self.lr_scheduler = \
            self.accelerator.prepare(model, self.opt, self.lr_scheduler)
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.writer = SummaryWriter(results_folder)
        self.results_folder = Path(results_folder)
        resume_file = str(self.results_folder / f'model-{resume_milestone}.pt')
        if os.path.isfile(resume_file):
            self.load(resume_milestone)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        if self.enable_resume:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt': self.opt.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
            # data_only_model = {'ema': self.ema.state_dict(),}
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        else:
            data = {
                'model': self.accelerator.get_state_dict(self.model),
            }
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        assert self.enable_resume; 'resume is available only if self.enable_resume is True !'
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.lr_scheduler.load_state_dict(data['lr_scheduler'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                total_loss_dict = {'loss_simple': 0., 'loss_vlb': 0., 'total_loss': 0., 'lr': 5e-5}
                for ga_ind in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)
                    batch = next(self.dl)
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key].to(device)
                    if self.step == 0 and ga_ind == 0:
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            self.model.module.on_train_batch_start(batch)
                        else:
                            self.model.on_train_batch_start(batch)

                    with self.accelerator.autocast():
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            loss, log_dict = self.model.module.training_step(batch)
                        else:
                            loss, log_dict = self.model.training_step(batch)

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        loss_simple = log_dict["train/loss_simple"] / self.gradient_accumulate_every
                        loss_vlb = log_dict["train/loss_vlb"] / self.gradient_accumulate_every
                        total_loss_dict['loss_simple'] += loss_simple
                        total_loss_dict['loss_vlb'] += loss_vlb
                        total_loss_dict['total_loss'] += total_loss
                        # total_loss_dict['s_fact'] = self.model.module.scale_factor
                        # total_loss_dict['s_bias'] = self.model.module.scale_bias

                    self.accelerator.backward(loss)
                total_loss_dict['lr'] = self.opt.param_groups[0]['lr']
                describtions = dict2str(total_loss_dict)
                describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                if accelerator.is_main_process:
                    pbar.desc = describtions

                if self.step % self.log_freq == 0:
                    if accelerator.is_main_process:
                        # pbar.desc = describtions
                        # self.logger.info(pbar.__str__())
                        self.logger.info(describtions)

                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)
                # pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.lr_scheduler.step()
                if accelerator.is_main_process:
                    self.writer.add_scalar('Learning_Rate', self.opt.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('total_loss', total_loss, self.step)
                    self.writer.add_scalar('loss_simple', loss_simple, self.step)
                    self.writer.add_scalar('loss_vlb', loss_vlb, self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                # if self.step >= int(self.train_num_steps * 0.2):
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        self.model.eval()
                        # self.ema.ema_model.eval()

                        with torch.no_grad():
                            # img = self.dl
                            # batches = num_to_groups(self.num_samples, self.batch_size)
                            # all_images_list = list(map(lambda n: self.model.module.validate_img(ns=self.batch_size), batches))
                            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                # all_images = self.model.module.sample(batch_size=self.batch_size)
                                if 'cond' in batch.keys():
                                    all_images, *_ = self.model.module.sample(batch_size=batch['cond'].shape[0],
                                                    cond=batch['cond'],
                                                    mask=batch['ori_mask'] if 'ori_mask' in batch else None)
                                else:
                                    all_images, *_ = self.model.module.sample(batch_size=self.batch_size)
                                
                            elif isinstance(self.model, nn.Module):
                                # all_images = self.model.sample(batch_size=self.batch_size)
                                all_images, *_ = self.model.sample(batch_size=batch['cond'].shape[0],
                                                  cond=batch['cond'],
                                                  mask=batch['ori_mask'] if 'ori_mask' in batch else None)
                                # pred = 
                                # print("=========================================")
                            # all_images = torch.clamp((all_images + 1.0) / 2.0, min=0.0, max=1.0)

                        # all_images = torch.cat(all_images_list, dim = 0)
                        # nrow = 2 ** math.floor(math.log2(math.sqrt(self.batch_size)))

                        # import torch
                        import math

                        def compute_brightest_point_distance(first_image, other_image):
                            """
                            计算两个单通道图像中最亮点的欧几里得距离。

                            参数:
                                first_image: torch.Tensor，形状为 [1, 256, 256]
                                other_image: torch.Tensor，形状为 [1, 256, 256]

                            返回:
                                float: 两个最亮点之间的欧几里得距离
                            """
                            # 去掉通道维度
                            first_image_2d = first_image.squeeze(0)  # [256, 256]
                            other_image_2d = other_image.squeeze(0)  # [256, 256]

                            # 找 first_image 的最亮点位置
                            first_max_idx = torch.argmax(first_image_2d)
                            first_y, first_x = divmod(first_max_idx.item(), first_image_2d.shape[1])

                            # 找 other_image 的最亮点位置
                            other_max_idx = torch.argmax(other_image_2d)
                            other_y, other_x = divmod(other_max_idx.item(), other_image_2d.shape[1])

                            # 计算欧几里得距离
                            distance = math.sqrt((first_x - other_x) ** 2 + (first_y - other_y) ** 2)
                            return distance

                        
                        import torchvision.utils as vutils

                        gt = batch['image'] #torch.Size([8, 1, 256, 256])
                        first_image = gt[0]
                        first_image = (first_image + 1) / 2
                        distance = compute_brightest_point_distance(first_image, all_images)
                        print(f"最亮点之间的距离: {distance}")
                        vutils.save_image(first_image, str(self.results_folder / f'gt-sample-{milestone}.png'))
                        # print(gt.shape)
                        # print(all_images.shape)

                        nrow = 2 ** math.floor(math.log2(math.sqrt(batch['cond'].shape[0])))
                        tv.utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=nrow)
                        self.model.train()
                accelerator.wait_for_everyone()
                pbar.update(1)

        accelerator.print('training complete')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass