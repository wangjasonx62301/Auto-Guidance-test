import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# -----------------------------------------------------------------------------
# Import your UNet and dataloader
from unet import UNet  # Ensure this matches your actual file structure
from dataset import get_dataloaders  # Ensure this matches your actual file structure
# Replace the following with your actual module paths
# from your_unet_file import UNet
# from your_dataloader_file import get_dataloaders
# -----------------------------------------------------------------------------

# --- PLACEHOLDER IMPORTS (delete these and uncomment your real imports) ---
# Minimal UNet stub so this file is syntactically valid if imported alone.
# You should REMOVE this stub and import your own UNet implementation.
# class UNet(nn.Module):
#     def __init__(self, in_ch=3, base_ch=64, ch_mults=(1, 2, 4), time_emb_dim=128, with_attn=(False, True, False)):
#         super().__init__()
#         raise NotImplementedError("Please import your actual UNet implementation and remove the stub.")

# Similarly, a stub for get_dataloaders; replace with your own.
# def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
#     raise NotImplementedError("Please import your actual get_dataloaders implementation and remove the stub.")
# -----------------------------------------------------------------------------


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DDPMConfig:
    image_size: int = 32
    channels: int = 3
    timesteps: int = 1000
    beta_schedule: str = "linear"  # [linear|cosine]
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    # training
    lr: float = 2e-4
    batch_size: int = 128
    epochs: int = 200
    grad_accum: int = 1
    ema_decay: float = 0.999
    logdir: str = "runs/ddpm"
    ckpt_every: int = 5

    # CFG
    num_classes: int = 10
    drop_cond_prob: float = 0.1    
    guidance_scale: float = 3.0    
    
# ---------------------------- Beta Schedules -------------------------------

def make_beta_schedule(T: int, schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":
        # Nichol & Dhariwal cosine schedule
        def f(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        ts = torch.linspace(0, 1, T + 1)
        alphas_bar = torch.tensor([f(t) for t in ts])
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(1e-8, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas


# ----------------------------- DDPM Core -----------------------------------

class DDPM(nn.Module):
    def __init__(self, model: nn.Module, config: DDPMConfig, device: torch.device = None):
        super().__init__()
        self.model = model
        self.cfg = config
        self.device = device or default_device()

        # precompute buffers
        betas = make_beta_schedule(config.timesteps, config.beta_schedule, config.beta_start, config.beta_end).to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_bar[:-1]]).to(device)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))

    # q(x_t | x_0)
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0).to(self.device)
        sqrt_ab = self.sqrt_alphas_bar[t][:, None, None, None].to(self.device)
        sqrt_omab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None].to(self.device)
        return sqrt_ab * x0 + sqrt_omab * noise

    # Training loss: predict epsilon (noise)
    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0).to(self.device)
        x_noisy = self.q_sample(x0, t, noise).to(self.device)
        
        if (y is not None) and (torch.rand(1, device=x0.device).item() < self.cfg.drop_cond_prob):
            y_in = None
        else:
            y_in = y
            
        noise_pred = self.model(x_noisy, t, y_in).to(self.device)
        loss = F.mse_loss(noise_pred, noise)
        return loss, noise_pred

    # Sampling step: x_{t-1} from x_t
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, guidance_scale: float = 1.0) -> torch.Tensor:
        betas_t = self.betas[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]

        # predict noise using the model
        if y is not None and guidance_scale == 1.0:
            # Classifier-Free Guidance
            eps_theta = self.model(x_t, t, None if y is None else y)
        else:
            eps_uncond = self.model(x_t, t, None)
            eps_cond   = self.model(x_t, t, y)
            eps_theta  = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDPM mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * eps_theta / sqrt_one_minus_alphas_bar_t)

        if (t == 0).all():
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            var = self.posterior_variance[t][:, None, None, None]
            return model_mean + torch.sqrt(var) * noise

    # Full sampling loop from pure noise
    @torch.no_grad()
    def sample(self, batch_size: int, labels: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        self.model.eval()
        g = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        img = torch.randn(batch_size, self.cfg.channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
        
        if labels is not None:
            labels = labels.to(self.device)
        
        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, y=labels, guidance_scale=g)
        return img


# ----------------------------- EMA Wrapper ---------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[name]
        self.backup = {}


# ----------------------------- Training Loop -------------------------------

@torch.no_grad()
def save_samples(ddpm: DDPM, out_dir: str, step: int, n: int = 16):
    os.makedirs(out_dir, exist_ok=True)
    samples = ddpm.sample(n)
    # The training data is normalized to [-1, 1]; save_image can auto-normalize
    save_path = os.path.join(out_dir, f"samples_step_{step:07d}.png")
    save_image(samples, save_path, nrow=int(math.sqrt(n)), normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {save_path}")


def train(cfg: DDPMConfig):
    device = default_device()

    # Data
    train_loader, _ = get_dataloaders(data_dir=os.path.join(cfg.logdir, "data"), batch_size=cfg.batch_size)

    # Model
    model = UNet(in_ch=cfg.channels, base_ch=128, ch_mults=(1, 2, 2, 2), time_emb_dim=512, with_attn=(False, True, True, False), num_classes=cfg.num_classes)
    model.to(device)

    # DDPM wrapper
    ddpm = DDPM(model, cfg, device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ema = EMA(model, decay=cfg.ema_decay)

    global_step = 0
    model.train()

    for epoch in range(cfg.epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device) if y is not None else None
            
            b = x.size(0)
            t = torch.randint(0, cfg.timesteps, (b,), device=device, dtype=torch.long)

            use_uncond = torch.rand(1, device=x.device).item() < cfg.drop_cond_prob
            y_in = None if use_uncond else y
            
            loss, _ = ddpm.p_losses(x, t, y=y_in)
            loss.backward()

            if (global_step + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)

            if global_step % 500 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")

            if global_step % 5000 == 0 and global_step > 0:
                # Save samples with EMA weights for better quality
                ema.apply_shadow(model)
                save_samples(ddpm, os.path.join(cfg.logdir, "samples"), step=global_step, n=16)
                ema.restore(model)

            global_step += 1

        # checkpoint per epoch
        if (epoch + 1) % cfg.ckpt_every == 0:
            os.makedirs(cfg.logdir, exist_ok=True)
            ckpt_path = os.path.join(cfg.logdir, f"ckpt_epoch_{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow,
                "opt": opt.state_dict(),
                "cfg": cfg.__dict__,
                "step": global_step,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


# ----------------------------- CLI -----------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Train DDPM on CIFAR-10 with your UNet")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--logdir", type=str, default="runs/ddpm")
    p.add_argument("--ckpt_every", type=int, default=5)
    
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--drop_cond_prob", type=float, default=0.1)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    return p


def main():
    cfg = DDPMConfig(**vars(build_argparser().parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()
