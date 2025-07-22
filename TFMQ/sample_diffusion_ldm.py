import argparse, os, sys, glob, datetime, yaml
from ldm.models.diffusion.plms import PLMSSampler
import math
from quant.reconstruction_util import RLOSS
from quant.data_generate import generate_cali_data_ldm
from quant.calibration import cali_model, cali_model_multi, load_cali_model
from quant.quant_model import QuantModel
from quant.quant_layer import QMODE, Scaler
import logging
import torch
import time
import numpy as np
from tqdm import trange
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image

import sys
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0, use_correction=False
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    if not use_correction:
        samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False)
    else:
        samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,
                                             correct=True, slope=torch.load('./data/slope.pt'), bias=torch.load('./data/bias.pt'),
                                             residual_error=torch.load('./data/residual_error.pt'))
    return samples, intermediates


@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0
                    ):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def convsample_plms(model, steps, shape, eta=1.0):
    plms = PLMSSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = plms.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False, use_correction=False, plms=False):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    # with model.ema_scope("Plotting"):
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    elif dpm:
        logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape,
                                                eta=eta)
    elif plms:
        logger.info(f'Using PLMS sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_plms(model,  steps=custom_steps, shape=shape,
                                                eta=eta)
    else:
        sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                eta=eta, use_correction=use_correction)

    t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    logger.info(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, nplog=None, dpm=False, use_correction=False, plms=False):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        logger.info(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(math.ceil(n_samples / batch_size), desc="Sampling Batches (unconditional)"):
            if (_ + 1) * batch_size > n_samples:
                assert _ == math.ceil(n_samples / batch_size) - 1
                batch_size = n_samples - _ * batch_size
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, dpm=dpm, use_correction=use_correction, plms=plms)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                logger.info(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        required=True,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampler",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    # linear quantization configs
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
    # qdiff specific configs
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--softmax_a_bit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--dpm", action="store_true",
        help="use dpm solver for sampling"
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
        default="cali_ckpt/quant_ddim.pth",
        help="path to save the calibrated ckpt"
    )
    parser.add_argument(
        "--interval_length",
        type=int,
        default=1,
        help="calibration interval length"
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
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


# def load_model(config, ckpt, gpu, eval_mode):
#     print(f"Loading model from {ckpt}")
#     if ckpt:
#         logger.info(f"Loading model from {ckpt}")
#         pl_sd = torch.load(ckpt, map_location="cpu")
#         #global_step = pl_sd["global_step"]
#         global_step = 50
#     else:
#         pl_sd = {"state_dict": None}
#         global_step = None
#     model = load_model_from_config(config.model,
#                                    pl_sd["state_dict"])

#     return model, global_step
from fvcore.common.config import CfgNode

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

def load_model(config, ckpt, gpu, eval_mode):
    print(f"Loading model from {ckpt}")
    ckpt = load_conf(ckpt)
    cfg = CfgNode(ckpt)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )
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
    from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
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

    global_step = 50
    return ldm, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # fix random seed
    seed_everything(opt.seed)
    ngpus_per_node = torch.cuda.device_count()
    opt.world_size = ngpus_per_node * opt.world_size

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    base_configs = '/home/Users_Work_Space/taokeda/projects/RadioDiff_V2/configs/BSDS_sample.yaml'
    opt.base = base_configs

    #configs = [OmegaConf.load(cfg) for cfg in opt.base]
    # cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli)
    config = OmegaConf.load(base_configs)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

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
    print(config)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"global step: {global_step}")
    logger.info("Switched to EMA weights")
    # model.model_ema.store(model.model.parameters())
    # model.model_ema.copy_to(model.model)


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
            setattr(model.model.diffusion_model, "split", True)
            qnn = QuantModel(model=model.model.diffusion_model,
                             wq_params=wq_params,
                             aq_params=aq_params,
                             cali=False,
                             softmax_a_bit=opt.softmax_a_bit,
                             aq_mode=opt.q_mode)
            qnn.to('cuda')
            qnn.eval()
            image_size = config.model.params.image_size
            channels = config.model.params.channels
            cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
            load_cali_model(qnn, cali_data, use_aq=opt.use_aq, path=opt.cali_ckpt)
            model.model.diffusion_model = qnn
            if opt.use_aq:
                cali_ckpt = torch.load(opt.cali_ckpt)
                tot = len(list(cali_ckpt.keys())) - 1
                model.model.tot = 1000 // tot
                model.model.t_max = tot - 1
                model.model.ckpt = cali_ckpt
                model.model.iter = 0
        else:
            logger.info("Generating calibration data...")
            shape = [config.params.channels, config.params.image_size, config.params.image_size]
            print(shape)
            model.device = 'cuda'
            qnn = QuantModel(model=model.model,
                             wq_params=wq_params,
                             aq_params=aq_params,
                             softmax_a_bit=opt.softmax_a_bit,
                             aq_mode=opt.q_mode)
            cali_data = generate_cali_data_ldm(model=model,
                                               T=opt.custom_steps,
                                               c=1,
                                               batch_size=256,
                                               shape=shape,
                                               vanilla=opt.vanilla_sample,
                                               dpm=opt.dpm,
                                               plms=opt.plms,
                                               eta=opt.eta)
            a_cali_data = cali_data
            tmp = []
            for i in range(0, opt.custom_steps, opt.interval_length):
                tmp.append([cali_data[0][i * 256: (i + 1) * 256], cali_data[1][i * 256: (i + 1) * 256]])
            w_cali_data = [torch.cat([x[0] for x in tmp]), torch.cat([x[1] for x in tmp])]
            logger.info("Calibration data generated.")
            torch.cuda.empty_cache()
            setattr(model.model.diffusion_model, "split", True)
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
                                                 256,
                                                 opt.running_stat,
                                                 kwargs), nprocs=ngpus_per_node)
            else:
                qnn = QuantModel(model=model.model.diffusion_model,
                                 wq_params=wq_params,
                                 aq_params=aq_params,
                                 softmax_a_bit=opt.softmax_a_bit,
                                 aq_mode=opt.q_mode)
                kwargs = dict(w_cali_data=w_cali_data,
                              a_cali_data=a_cali_data,
                              iters=20000,
                              batch_size=32,
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
                           interval=256,
                           **kwargs)
            exit(0)

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        print(sampling_conf)
        logger.info("first_stage_model")
        logger.info(model.first_stage_model)
        logger.info("UNet model")
        logger.info(model.model)

    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir, dpm=opt.dpm, use_correction=QMODE.PTQD.value in opt.q_mode, plms=opt.plms)

    logger.info("done.")
