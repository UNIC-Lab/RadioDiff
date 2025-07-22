import math
import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

import sys
from ..models.diffusion import Model
from ..models.ema import EMAHelper
from ..functions import get_optimizer
from ..functions.losses import loss_registry
from ..datasets import get_dataset, data_transform, inverse_data_transform
from ..functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
from quant.calibration import cali_model, load_cali_model
from quant.data_generate import generate_cali_data_ddim

from quant.quant_layer import QMODE, Scaler
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS
import logging
logger = logging.getLogger(__name__)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        config.split_shortcut = True
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            # model = torch.nn.DataParallel(model)
            model.eval()

        # ------------ quantize model ------------ #
        tot, cali_ckpt, t_max = None, None, None
        if self.args.ptq:
            wq_params = {"bits": self.args.wq,
                         "channel_wise": True,
                         "scaler": Scaler.MINMAX if not self.args.cali else Scaler.MSE}
            aq_params = {"bits": self.args.aq,
                         "channel_wise": False,
                         "scaler": Scaler.MINMAX if not self.args.cali else Scaler.MSE,
                         "leaf_param": self.args.use_aq}
            if not self.args.cali:
                qnn = QuantModel(model=model,
                                 wq_params=wq_params,
                                 aq_params=aq_params,
                                 cali=False,
                                 softmax_a_bit=self.args.softmax_a_bit,
                                 aq_mode=self.args.q_mode)
                qnn.to(self.device)
                qnn.eval()
                image_size = self.config.data.image_size
                channels = self.config.data.channels
                cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
                load_cali_model(qnn, cali_data, use_aq=self.args.use_aq, path=self.args.cali_ckpt)
                model = qnn
                if self.args.use_aq:
                    cali_ckpt = torch.load(self.args.cali_ckpt)
                    tot = 1000 - (len(list(cali_ckpt.keys())) - 1)
                    t_max = len(list(cali_ckpt.keys())) - 2
            else:
                logger.info("Generating calibration data...")
                cali_data = generate_cali_data_ddim(runnr=self,
                                                    model=model,
                                                    T= self.args.timesteps,
                                                    c=1,
                                                    batch_size=256,
                                                    shape=(self.config.data.channels,
                                                           self.config.data.image_size,
                                                           self.config.data.image_size,))
                a_cali_data = cali_data
                tmp = []
                for i in range(0, self.args.timesteps, self.args.interval_length):
                    tmp.append([cali_data[0][i * 256: (i + 1) * 256], cali_data[1][i * 256: (i + 1) * 256]])
                w_cali_data = [torch.cat([x[0] for x in tmp], dim=0), torch.cat([x[1] for x in tmp], dim=0)]
                logger.info("Calibration data generated.")
                qnn = QuantModel(model=model,
                                 wq_params=wq_params,
                                 aq_params=aq_params,
                                 softmax_a_bit=self.args.softmax_a_bit,
                                 aq_mode=self.args.q_mode)
                qnn.to(self.device)
                qnn.eval()
                kwargs = dict(w_cali_data=w_cali_data,
                              a_cali_data=a_cali_data,
                              iters=20000,
                              batch_size=32,
                              w=0.01,
                              asym=self.args.asym,
                              warmup=0.2,
                              opt_mode=RLOSS.MSE,
                              multi_gpu=False)
                cali_model(qnn=qnn,
                           use_aq=self.args.use_aq,
                           path=self.args.cali_save_path,
                           running_stat=self.args.running_stat,
                           interval=256,
                           **kwargs)
                model = qnn
                exit(0)

        model.to(self.device)
        model.eval()

        if self.args.fid:
            self.sample_fid(model, tot=tot, cali_ckpt=cali_ckpt, t_max=t_max)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, untill_fake_t=114514, tot=None, cali_ckpt=None, t_max=None):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        logger.info(f"starting from image {img_id}")
        total_n_samples = self.args.max_images
        n_rounds = math.ceil((total_n_samples - img_id) / config.sampling.batch_size)
        with torch.no_grad():
            all_results = []
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, untill_fake_t=untill_fake_t, tot=tot, cali_ckpt=cali_ckpt, t_max=t_max)[0]
                x = inverse_data_transform(config, x)
                if img_id + x.shape[0] > total_n_samples:
                    assert _ == n_rounds - 1
                    n = self.args.max_images - img_id
                    np_x_rgb = x[:n].permute(0, 2, 3, 1).cpu().numpy() * 255.
                else:
                    np_x_rgb = x.permute(0, 2, 3, 1).cpu().numpy() * 255.
                np_x_rgb = np_x_rgb.round().astype(np.uint8)
                all_results.append(np_x_rgb)
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1
            all_results = np.concatenate(all_results[:total_n_samples], axis=0)
            shape_str = "x".join([str(x) for x in all_results.shape])
            nppath = os.path.join(self.args.numpy_folder, f'{shape_str}-samples.npz')
            np.savez(nppath, all_results)

    def sample_sequence(self, model, untill_fake_t=114514):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, untill_fake_t=untill_fake_t)[0]

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model, untill_fake_t=114514):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model, untill_fake_t=untill_fake_t)[0])
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True, untill_fake_t=114514, tot=None, cali_ckpt=None, t_max=None):
        x_t, t_t = None, None
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ..functions.denoising import generalized_steps

            xs, x0_preds, x_t, t_t = generalized_steps(x, seq, model, self.betas, eta=self.args.eta, untill_fake_t=untill_fake_t, tot=tot, cali_ckpt=cali_ckpt, t_max=t_max)
            x = (xs, x0_preds)
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x, x0_preds, x_t, t_t = ddpm_steps(x, seq, model, self.betas, untill_fake_t=untill_fake_t, tot=tot, cali_ckpt=cali_ckpt, t_max=t_max)
            x = (x, x0_preds)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x, x_t, t_t

    def test(self):
        pass
