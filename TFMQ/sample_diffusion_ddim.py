import argparse
import datetime
import logging
import os
from pytorch_lightning import seed_everything

import yaml

from ddim.runners.diffusion import Diffusion
from quant.quant_layer import QMODE


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")

    # quantization configs
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
        "--max_images", type=int, default=50000, help="number of images to sample"
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
    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # setup logger
    logdir = os.path.join(args.logdir, "samples", now)
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
    nplogdir = os.path.join(logdir, "numpy")
    os.makedirs(nplogdir)
    args.image_folder = imglogdir
    args.numpy_folder = nplogdir

    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")
    p = [QMODE.NORMAL.value]
    p.append(QMODE.QDIFF.value)
    args.q_mode = p
    args.fid = True
    args.log_path = "test/"
    args.use_pretrained = True
    args.use_aq = args.use_aq
    args.asym = True
    args.running_stat = True
    config.device = 'cuda0'
    runner = Diffusion(args, config)
    runner.sample()

