import torch
import torch.nn as nn
import sys
# .path.append()
from taming.modules.losses.vqperceptual import *


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, *, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) + \
                    F.mse_loss(inputs, reconstructions, reduction="none")
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

class LPIPSWithDiscriminator_Edge(nn.Module):
    def __init__(self, *, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, k2_neg_norm, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) + \
                    F.mse_loss(inputs, reconstructions, reduction="none")
        inputs_pinn = inputs * (1. - k2_neg_norm)
        reconstructions_pinn = reconstructions * (1. - k2_neg_norm)
        pinn_loss = torch.abs(inputs_pinn.contiguous() - reconstructions_pinn.contiguous()) + \
                    F.mse_loss(inputs_pinn, reconstructions_pinn, reduction="none")
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            pinn_loss = pinn_loss + self.perceptual_weight * p_loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_pinn_loss = pinn_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        weighted_nll_pinn_loss = nll_pinn_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
            weighted_nll_pinn_loss = weights*nll_pinn_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        weighted_nll_pinn_loss = torch.sum(weighted_nll_pinn_loss) / weighted_nll_pinn_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_pinn_loss = torch.sum(nll_pinn_loss) / nll_pinn_loss.shape[0]
        
        nll_loss_total = nll_loss + 2.5 * nll_pinn_loss
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    # d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    d_weight = self.calculate_adaptive_weight(nll_loss_total, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    # d_weight = torch.tensor(0.0)
                    d_weight = torch.tensor(0.0)
            else:
                # d_weight = torch.tensor(0.0)
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + weighted_nll_pinn_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/pinn_loss".format(split): pinn_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
        




class LPIPSWithDiscriminator_Edge_PINN(nn.Module):
    def __init__(self, *, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
        self.PathLoss_trunc = -147 # db
        self.PathLoss_max = -47 # db
        self.source_power = 23 # dbm
        self.h = 1.00
        self.one_over_h2 = 1 / self.h**2
        self.source_power_db = self.source_power - 30
        
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions,  posteriors, Tx_pos_mask, k2_neg_norm, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) + \
                    F.mse_loss(inputs, reconstructions, reduction="none")
        inputs_edge = inputs * (1. - k2_neg_norm)
        reconstructions_edge = reconstructions * (1. - k2_neg_norm)
        edge_loss = torch.abs(inputs_edge.contiguous() - reconstructions_edge.contiguous()) + \
                    F.mse_loss(inputs_edge, reconstructions_edge, reduction="none")
        
        # loss pinn
        inputs_PathLoss_scale = inputs
        reconstructions_PathLoss_scale = reconstructions
        inputs_PathLoss_db = self.PathLoss_trunc + (self.PathLoss_max - self.PathLoss_trunc) * inputs_PathLoss_scale
        reconstructions_PathLoss_db = self.PathLoss_trunc + (self.PathLoss_max - self.PathLoss_trunc) * reconstructions_PathLoss_scale
        
        inputs_power_db = torch.ones_like(inputs_PathLoss_db) * self.source_power_db + inputs_PathLoss_db
        reconstructions_power_db = torch.ones_like(reconstructions_PathLoss_db) * self.source_power_db + reconstructions_PathLoss_db
        
        inputs_power_w = 10**(inputs_power_db / 10)
        reconstructions_power_w = 10**(reconstructions_power_db / 10)
        
        inputs_u = inputs_power_w
        inputs_re = reconstructions_power_w
        
        inputs_delta_u = (inputs_u[:, :, 2:, 1:-1] + inputs_u[:, :, :-2, 1:-1] + inputs_u[:, :, 1:-1, 2:] + inputs_u[:, :, 1:-1, :-2]) - 4 * inputs_u[:, :, 1:-1, 1:-1]
        inputs_delta_re = (inputs_re[:, :, 2:, 1:-1] + inputs_re[:, :, :-2, 1:-1] + inputs_re[:, :, 1:-1, 2:] + inputs_re[:, :, 1:-1, :-2]) - 4 * inputs_re[:, :, 1:-1, 1:-1]
        
        inputs_delta_u = inputs_delta_u * self.one_over_h2
        inputs_delta_re = inputs_delta_re * self.one_over_h2
        # print("inputs_delta_u.shape", inputs_delta_u.shape)
        # print("Tx_pos_mask.shape", Tx_pos_mask.shape)
        # print("inputs_u.shape", inputs_u.shape)
        Tx_pos_mask = Tx_pos_mask.unsqueeze(1)
        inputs_k2 = (inputs_delta_u + Tx_pos_mask[:, :, 1:-1, 1:-1] * inputs_u[:, :, 1:-1, 1:-1]) / inputs_u[:, :, 1:-1, 1:-1]
        reconstructions_k2 = (inputs_delta_re + Tx_pos_mask[:, :, 1:-1, 1:-1] * inputs_re[:, :, 1:-1, 1:-1]) / inputs_re[:, :, 1:-1, 1:-1]
        
        inputs_k2_norm = torch.norm(inputs_k2, dim=1)
        
        pinn_loss = F.mse_loss(inputs_k2.contiguous(), reconstructions_k2.contiguous(), reduction="none")
        pinn_loss = torch.clamp(pinn_loss, min=1e-7, max=0.2-1e-7)
        
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            edge_loss = edge_loss + self.perceptual_weight * p_loss
            pinn_loss = pinn_loss + self.perceptual_weight * p_loss
            
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_edge_loss = edge_loss / torch.exp(self.logvar) + self.logvar
        nll_pinn_loss = pinn_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        weighted_nll_edge_loss = nll_edge_loss
        weighted_nll_pinn_loss = nll_pinn_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
            weighted_nll_edge_loss = weights*nll_edge_loss
            weighted_nll_pinn_loss = weights*nll_pinn_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        weighted_nll_edge_loss = torch.sum(weighted_nll_edge_loss) / weighted_nll_edge_loss.shape[0]
        weighted_nll_pinn_loss = torch.sum(weighted_nll_pinn_loss) / weighted_nll_pinn_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_edge_loss = torch.sum(nll_edge_loss) / nll_edge_loss.shape[0]
        nll_pinn_loss = torch.sum(nll_pinn_loss) / nll_pinn_loss.shape[0]
        nll_loss_total = nll_loss + 1.5 * nll_edge_loss
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    # d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    d_weight = self.calculate_adaptive_weight(nll_loss+nll_edge_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    # d_weight = torch.tensor(0.0)
                    d_weight = torch.tensor(0.0)
            else:
                # d_weight = torch.tensor(0.0)
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + weighted_nll_edge_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/edge_loss".format(split): edge_loss.detach().mean(),
                   "{}/pinn_loss".format(split): pinn_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class LPIPSWithDiscriminator_DPM2IRT4(nn.Module):
    def __init__(self, *, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
        self.PathLoss_trunc = -147 # db
        self.PathLoss_max = -47 # db
        self.source_power = 23 # dbm
        self.h = 1.00
        self.one_over_h2 = 1 / self.h**2
        self.source_power_db = self.source_power - 30
        
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions,  posteriors, Tx_pos_mask, k2_neg_norm, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) + \
                    F.mse_loss(inputs, reconstructions, reduction="none")
        inputs_edge = inputs * (1. - k2_neg_norm)
        reconstructions_edge = reconstructions * (1. - k2_neg_norm)
        edge_loss = torch.abs(inputs_edge.contiguous() - reconstructions_edge.contiguous()) + \
                    F.mse_loss(inputs_edge, reconstructions_edge, reduction="none")
        
        # loss pinn
        inputs_PathLoss_scale = inputs
        reconstructions_PathLoss_scale = reconstructions
        inputs_PathLoss_db = self.PathLoss_trunc + (self.PathLoss_max - self.PathLoss_trunc) * inputs_PathLoss_scale
        reconstructions_PathLoss_db = self.PathLoss_trunc + (self.PathLoss_max - self.PathLoss_trunc) * reconstructions_PathLoss_scale
        
        inputs_power_db = torch.ones_like(inputs_PathLoss_db) * self.source_power_db + inputs_PathLoss_db
        reconstructions_power_db = torch.ones_like(reconstructions_PathLoss_db) * self.source_power_db + reconstructions_PathLoss_db
        
        inputs_power_w = 10**(inputs_power_db / 10)
        reconstructions_power_w = 10**(reconstructions_power_db / 10)
        
        inputs_u = inputs_power_w
        inputs_re = reconstructions_power_w
        
        inputs_delta_u = (inputs_u[:, :, 2:, 1:-1] + inputs_u[:, :, :-2, 1:-1] + inputs_u[:, :, 1:-1, 2:] + inputs_u[:, :, 1:-1, :-2]) - 4 * inputs_u[:, :, 1:-1, 1:-1]
        inputs_delta_re = (inputs_re[:, :, 2:, 1:-1] + inputs_re[:, :, :-2, 1:-1] + inputs_re[:, :, 1:-1, 2:] + inputs_re[:, :, 1:-1, :-2]) - 4 * inputs_re[:, :, 1:-1, 1:-1]
        
        inputs_delta_u = inputs_delta_u * self.one_over_h2
        inputs_delta_re = inputs_delta_re * self.one_over_h2
        # print("inputs_delta_u.shape", inputs_delta_u.shape)
        # print("Tx_pos_mask.shape", Tx_pos_mask.shape)
        # print("inputs_u.shape", inputs_u.shape)
        Tx_pos_mask = Tx_pos_mask.unsqueeze(1)
        inputs_k2 = inputs_delta_u / (inputs_u[:, :, 1:-1, 1:-1] + 1e-9)
        # reconstructions_k2 = (inputs_delta_re + Tx_pos_mask[:, :, 1:-1, 1:-1] * inputs_re[:, :, 1:-1, 1:-1]) / inputs_re[:, :, 1:-1, 1:-1]
        pinn_loss = torch.zeros_like(inputs_u) + 1e-9
        r = (inputs_delta_re - inputs_k2 * inputs_re[:, :, 1:-1, 1:-1]) * (1 - Tx_pos_mask[:, :, 1:-1, 1:-1]) 
        pinn_loss[:, :, 1:-1, 1:-1] += torch.abs(r)
        
        # print("max pinn : ", torch.max(pinn_loss))
        # print("min pinn : ", torch.min(pinn_loss))
        # inputs_k2_norm = torch.norm(inputs_k2, dim=1)
        pinn_loss = torch.clamp(pinn_loss, min=1e-7, max=0.15-1e-7)
        
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            edge_loss = edge_loss + self.perceptual_weight * p_loss
            pinn_loss = pinn_loss + self.perceptual_weight * p_loss
            
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_edge_loss = edge_loss / torch.exp(self.logvar) + self.logvar
        nll_pinn_loss = pinn_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        weighted_nll_edge_loss = nll_edge_loss
        weighted_nll_pinn_loss = nll_pinn_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
            weighted_nll_edge_loss = 1.5 * weights*nll_edge_loss
            weighted_nll_pinn_loss = 2.0 * weights*nll_pinn_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        weighted_nll_edge_loss = torch.sum(weighted_nll_edge_loss) / weighted_nll_edge_loss.shape[0]
        weighted_nll_pinn_loss = torch.sum(weighted_nll_pinn_loss) / weighted_nll_pinn_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_edge_loss = torch.sum(nll_edge_loss) / nll_edge_loss.shape[0]
        nll_pinn_loss = torch.sum(nll_pinn_loss) / nll_pinn_loss.shape[0]
        nll_loss_total = nll_loss + nll_edge_loss + nll_pinn_loss
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    # d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    d_weight = self.calculate_adaptive_weight(nll_loss_total, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    # d_weight = torch.tensor(0.0)
                    d_weight = torch.tensor(0.0)
            else:
                # d_weight = torch.tensor(0.0)
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + weighted_nll_edge_loss + weighted_nll_pinn_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/edge_loss".format(split): edge_loss.detach().mean(),
                   "{}/pinn_loss".format(split): pinn_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
        
        
        
        
        
        
        
        
        
        
        
        
