from pytorch_lightning import seed_everything
from quant.calibration import cali_model, cali_model_multi, load_cali_model
from quant.data_generate import generate_cali_data_ldm_imagenet
from quant.quant_layer import QMODE, Scaler
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS
import sys
import os
import datetime
import argparse
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import logging


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("./stable-diffusion/configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "./stable-diffusion/models/ldm/cin256-v2/model.ckpt")
    return model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--n_sample_per_class",
        type=int,
        default=3,
        help="how many samples to produce for each given class. A.k.a. batch size",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0",
        help="comma-separated list of classes to sample from",
    )
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--wq",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--aq",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--softmax_a_bit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
       "--verbose", action="store_true",
       help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--cali",
        action="store_true",
        help="whether to calibrate the model"
    )
    parser.add_argument(
        "--cali_save_path",
        type=str,
        default="cali_ckpt/quant_sd.pth",
        help="path to save the calibrated ckpt"
    )
    parser.add_argument(
        "--cali_data_path",
        type=str,
        help="prompts data path"
    )
    parser.add_argument(
        "--interval_length",
        type=int,
        default=1,
        help="calibration interval length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=41,
        help="random seed"
    )
    parser.add_argument(
        '--use_aq',
        action='store_true',
        help='whether to use activation quantization'
    )
    # multi-gpu configs
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3367', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    return parser.parse_args()


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)
    
    opt = get_parser()
    seed_everything(opt.seed)
    ngpus_per_node = torch.cuda.device_count()
    opt.world_size = ngpus_per_node * opt.world_size
    logdir = os.path.join(opt.outdir, 'imagenet')
    logdir = os.path.join(logdir, "samples", now)
    os.makedirs(logdir)
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")
    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    logger.info(logdir)
    logger.info(75 * "=")
    model = get_model()
    sampler = DDIMSampler(model)

    ddim_steps = opt.ddim_steps
    ddim_eta = opt.eta
    scale = opt.scale   # for unconditional guidance
    
    
    p = [QMODE.NORMAL.value]
    p.append(QMODE.QDIFF.value)
    opt.q_mode = p
    opt.asym = True
    opt.running_stat = True
    wq_params = {"bits": opt.wq,
                 "channel_wise": True,
                 "scaler": Scaler.MSE if opt.cali else Scaler.MINMAX}
    aq_params = {"bits": opt.aq,
                 "channel_wise": False,
                 "scaler": Scaler.MSE if opt.cali else Scaler.MINMAX,
                 "leaf_param": opt.use_aq}
    if opt.ptq:
        if not opt.cali:
            setattr(sampler.model.model.diffusion_model, "split", True)
            qnn = QuantModel(model=sampler.model.model.diffusion_model,
                             wq_params=wq_params,
                             aq_params=aq_params,
                             cali=False,
                             softmax_a_bit=opt.softmax_a_bit,
                             aq_mode=opt.q_mode)
            qnn.to('cuda')
            qnn.eval()
            image_size = 64
            channels = 3
            uc_t = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(1 * [1000]).to(model.device)})
            cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)), uc_t)
            load_cali_model(qnn, cali_data, use_aq=opt.use_aq, path=opt.cali_ckpt)
            sampler.model.model.diffusion_model = qnn
            if opt.use_aq:
                cali_ckpt = torch.load(opt.cali_ckpt)
                tot = len(list(cali_ckpt.keys())) - 1
                sampler.model.model.tot = 1000 // tot
                model.model.t_max = tot - 1
                sampler.model.model.ckpt = cali_ckpt
                sampler.model.model.iter = 0
        else:
            logger.info("Generating calibration data...")
            shape = [3, 64, 64]
            cali_data = generate_cali_data_ldm_imagenet(model=model,
                                                        T=opt.ddim_steps,
                                                        c=1,
                                                        batch_size=8,
                                                        shape=shape,
                                                        eta=opt.eta,
                                                        scale=opt.scale)
            a_cali_data = cali_data
            w_cali_data = cali_data
            logger.info("Calibration data generated.")
            torch.cuda.empty_cache()
            setattr(sampler.model.model.diffusion_model, "split", True)
            if opt.multi_gpu:
                kwargs = dict(iters=20000,
                              batch_size=32,
                              w=0.01,
                              asym=opt.asym,
                              warmup=0.2,
                              opt_mode=RLOSS.MSE,
                              wq_params=wq_params,
                              aq_params=aq_params,
                              softmax_a_bit=opt.softmax_a_bit,
                              aq_mode=opt.q_mode,
                              multi_gpu=ngpus_per_node > 1)
                mp.spawn(cali_model_multi, args=(opt.dist_backend,
                                                 opt.world_size,
                                                 opt.dist_url,
                                                 opt.rank,
                                                 ngpus_per_node,
                                                 model.model,
                                                 opt.use_aq,
                                                 opt.cali_save_path,
                                                 w_cali_data,
                                                 a_cali_data,
                                                 512,
                                                 opt.running_stat,
                                                 kwargs), nprocs=ngpus_per_node)
            else:
                qnn = QuantModel(model=sampler.model.model.diffusion_model,
                                 wq_params=wq_params,
                                 aq_params=aq_params,
                                 softmax_a_bit=opt.softmax_a_bit,
                                 aq_mode=opt.q_mode)
                kwargs = dict(w_cali_data=w_cali_data,
                              a_cali_data=a_cali_data,
                              iters=20000,
                              batch_size=8,
                              w=0.01,
                              asym=opt.asym,
                              warmup=0.2,
                              opt_mode=RLOSS.MSE,
                              multi_gpu=False)
                qnn.to('cuda')
                qnn.eval()
                cali_model(qnn=qnn,
                           use_aq=opt.use_aq,
                           path=opt.cali_save_path,
                           running_stat=opt.running_stat,
                           interval=512,
                           **kwargs)
            exit(0)
    
    
    
    # all_samples = list()
    n_samples_per_class = opt.n_sample_per_class
    if opt.classes == "all":
        classes = list(range(1000))
    else:
        classes = [int(c) for c in opt.classes.split(",")]

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            base_count = 0
            all_imags = []
            all_labels = []
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(imglogdir + f"/{class_label}_{base_count:05f}.png")
                    base_count += 1
                # all_samples.append(x_samples_ddim)
                sample = 255. * rearrange(x_samples_ddim, 'b c h w -> b h w c').cpu().numpy()
                sample = sample.astype(np.uint8)
                all_imags.extend([sample])
                all_labels.extend([xc.cpu().numpy()])

    all_img = np.concatenate(all_imags, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    shape_str = "x".join([str(x) for x in all_img.shape])
    label_shape_str = "x".join([str(x) for x in all_labels.shape])
    nppath = os.path.join(numpylogdir, f"{shape_str}-{label_shape_str}-samples.npz")
    np.savez(nppath, all_img, all_labels)


    # display as grid
    # grid = torch.stack(all_samples, 0)
    # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    # grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # img = Image.fromarray(grid.astype(np.uint8))
    # img.save(logdir + f'/grid.png')