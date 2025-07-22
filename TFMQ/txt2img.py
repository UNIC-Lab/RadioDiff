import argparse, os, datetime, yaml
import math
import logging
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from quant.calibration import cali_model, cali_model_multi, load_cali_model
from quant.data_generate import generate_cali_text_guided_data

from quant.quant_layer import QMODE, Scaler
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS
import torch.multiprocessing as mp
import pandas as pd
import json

logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def get_prompts(path: str,
                num: int = 128):
    '''
        COCO-Captions dataset
    '''
    df = pd.DataFrame(json.load(open(path))['annotations'])
    ps = df['caption'].sample(num).tolist()
    return ps


def prompts4eval(path: str,
                 batch_size: int = 1):
    df = pd.read_parquet(path)
    prompts = df['caption'].tolist()
    res = []
    for i in range(math.ceil(len(prompts) / batch_size)):
        if (i + 1) * batch_size > len(prompts):
            res.append(prompts[i * batch_size:])
        else:
            res.append(prompts[i * batch_size:(i + 1) * batch_size])
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
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
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cond", action="store_true",
       help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
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
        "--data_path",
        type=str,
        default="",
        help="prompts data path(for calibration or evaluation)"
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
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    ngpus_per_node = torch.cuda.device_count()
    opt.world_size = ngpus_per_node * opt.world_size

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    log_path = os.path.join(outpath, "run.log")
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

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)


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
    assert(opt.cond)
    if opt.ptq:
        # LatentDiffsion --> Diffusionwrapper --> Unet
        if not opt.cali:
            setattr(sampler.model.model.diffusion_model, "split", True)
            qnn = QuantModel(model=sampler.model.model.diffusion_model,
                             wq_params=wq_params,
                             aq_params=aq_params,
                             cali=False,
                             softmax_a_bit=opt.softmax_a_bit,
                             aq_mode=opt.q_mode)
            qnn.to(device)
            qnn.eval()
            if opt.no_grad_ckpt:
                logger.info('Not use gradient checkpointing for transformer blocks')
                qnn.set_grad_ckpt(False)
            cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
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
            cali_data = generate_cali_text_guided_data(model,
                                                       sampler,
                                                       T=opt.ddim_steps,
                                                       c=1,
                                                       batch_size=1,
                                                       prompts=get_prompts(opt.data_path),
                                                       shape=[opt.C, opt.H // opt.f, opt.W // opt.f],
                                                       precision_scope=autocast if opt.precision=="autocast" else nullcontext)
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
                              multi_gpu=ngpus_per_node > 1,
                              no_grad_ckpt=opt.no_grad_ckpt)
                if opt.no_grad_ckpt:
                    logger.info('Not use gradient checkpointing for transformer blocks')
                mp.spawn(cali_model_multi, args=(opt.dist_backend,
                                                 opt.world_size,
                                                 opt.dist_url,
                                                 opt.rank,
                                                 ngpus_per_node,
                                                 sampler.model.model,
                                                 opt.use_aq,
                                                 opt.cali_save_path,
                                                 w_cali_data,
                                                 a_cali_data,
                                                 256,
                                                 opt.running_stat,
                                                 kwargs), nprocs=ngpus_per_node)
            else:
                qnn = QuantModel(model=sampler.model.model.diffusion_model,
                                 wq_params=wq_params,
                                 aq_params=aq_params,
                                 softmax_a_bit=opt.softmax_a_bit,
                                 aq_mode=opt.q_mode)
                qnn.to(device)
                qnn.eval()
                if opt.no_grad_ckpt:
                    logger.info('Not use gradient checkpointing for transformer blocks')
                    qnn.set_grad_ckpt(False)
                kwargs = dict(w_cali_data=w_cali_data,
                              a_cali_data = a_cali_data,
                              iters=20000,
                              batch_size=8,
                              w=0.01,
                              asym=opt.asym,
                              warmup=0.2,
                              opt_mode=RLOSS.MSE,
                              multi_gpu=False)
                cali_model(qnn=qnn,
                           use_aq=opt.use_aq,
                           path=opt.cali_save_path,
                           running_stat=opt.running_stat,
                           interval=256,
                           **kwargs)
            exit(0)

    logging.info("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        logging.info(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))
    if opt.data_path is not "":
        data = prompts4eval(opt.data_path, batch_size)
        opt.n_iter = 1
    sample_path = os.path.join(outpath, "imags")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(os.path.join(outpath, "texts"), exist_ok=True)
    os.makedirs(os.path.join(outpath, "numpy"), exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        logger.info("UNet model")
        logger.info(model.model)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                all_images= []
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                with open(os.path.join(outpath, 'texts', f"{base_count:05}.txt"), "w") as f:
                                    f.write(prompts[base_count % batch_size])
                                base_count += 1
                            imags = 255. * rearrange(x_checked_image_torch, 'n c h w -> n h w c').cpu().numpy()
                            imags = imags.astype(np.uint8)
                            all_images.extend([imags])

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

                all_img = np.concatenate(all_images, axis=0)
                shape_str = "x".join([str(x) for x in all_img.shape])
                nppath = os.path.join(outpath, 'numpy', f"{shape_str}-samples.npz")
                np.savez(nppath, all_img)

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
