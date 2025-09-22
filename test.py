import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm
import logging
from datetime import datetime
# -----------------------------------------------------------------------------
from unet import UNet 
from dataset import get_dataloaders  
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

# Set up logging
def setup_logging(logdir: str, filename: str = "training.log"):
    os.makedirs(logdir, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    log_path = os.path.join(logdir, filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file at {log_path}")


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DDPMConfig:
    image_size: int = 32
    channels: int = 3
    timesteps: int = 5000
    beta_schedule: str = "linear"  # [linear|cosine]
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    # training
    lr: float = 2e-4
    batch_size: int = 128
    epochs: int = 10000
    grad_accum: int = 1
    ema_decay: float = 0.999
    logdir: str = "runs/ddpm"
    ckpt_every: int = 5

    # CFG
    num_classes: int = 10
    drop_cond_prob: float = 0.1    
    guidance_scale: float = 3.0    
    guidance_mode: Optional[str] = 'cfg'  # 'cfg' or 'autog' or None
    train_with_autog: str = 'False'  # whether to use Auto-Guidance during training (requires bad model)
    fid_threshold: float = 2.0  # update bad model if fid improves by this much
    
    # path
    ckpt_path: Optional[str] = None
    bad_model_ckpt: Optional[str] = None  # path to bad model checkpoint for Auto-Guidance
    
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
    def __init__(self, model: nn.Module, config: DDPMConfig, device: torch.device = None, bad_model: nn.Module = None):
        super().__init__()
        self.model = model
        self.cfg = config
        self.device = device or default_device()
        self.bad_model = bad_model

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

    def get_bad_model_from_snapshot(self, bad_model_ckpt_path):
        self.bad_model = copy.deepcopy(self.model).eval().to(self.device)
        self.bad_model.load_state_dict(torch.load(bad_model_ckpt_path, map_location=self.device)['model'])
        logging.info(f"Loaded bad model from {bad_model_ckpt_path} for Auto-Guidance")
        for p in self.bad_model.parameters():
            p.requires_grad = False
    
    def update_bad_model(self, snapshot_state_dict):
        assert self.bad_model is not None, "Bad model is not set. Call get_bad_model_from_snapshot() first."
        self.bad_model.load_state_dict(snapshot_state_dict)
        self.bad_model.eval()
        for p in self.bad_model.parameters():
            p.requires_grad = False
            
    
    # q(x_t | x_0)
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0).to(self.device)
        sqrt_ab = self.sqrt_alphas_bar[t][:, None, None, None].to(self.device)
        sqrt_omab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None].to(self.device)
        return sqrt_ab * x0 + sqrt_omab * noise
    
    def q_posterior_mean(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        alpha_bar_t = self.alphas_bar[t][:, None, None, None].to(self.device)
        alpha_bar_prev = self.alphas_bar_prev[t][:, None, None, None].to(self.device)
        betas_t = self.betas[t][:, None, None, None].to(self.device)
        
        coef1 = betas_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(1.0 / (alpha_bar_t)) / (1.0 - alpha_bar_t)
        
        mu_q = coef1 * x0 + coef2 * x_t
        return mu_q

    # Training loss: predict epsilon (noise)
    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None, train_with_autog: Optional[str] = 'False', t_2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0).to(self.device)
        x_noisy = self.q_sample(x0, t, noise).to(self.device)
        
        if (y is not None) and (torch.rand(1, device=x0.device).item() < self.cfg.drop_cond_prob):
            y_in = None
        else:
            y_in = y
        
        # if self.bad_model is not None and train_with_autog == 'True':
            # print('Using Auto-Guidance during training')
        self.bad_model.eval()
        eps_pos = self.model(x_noisy, t_2, y_in)
        eps_bad = self.bad_model(x_noisy, t, None)
        w = guidance_scale
        eps_theta = eps_pos + w * (eps_pos - eps_bad)
        
        with torch.no_grad():
            eps_pos = self.model(x_noisy, t_2, None).detach()
            eps_bad = self.bad_model(x_noisy, t, None).detach()
            diff = (eps_pos - eps_bad).view(eps_pos.size(0), -1)
            l2_per_sample = (diff ** 2).sum(dim=1).cpu().numpy()
            logging.info(f"Auto-Guidance L2 norm per sample: {l2_per_sample.mean():.4f} ± {l2_per_sample.std():.4f}")
        
        # else:
            # print('Not using Auto-Guidance during training')
        # eps_theta = self.model(x_noisy, t, y_in).to(self.device)
        # noise_pred = self.model(x_noisy, t, y_in).to(self.device)
        # loss = F.mse_loss(eps_theta, noise, reduction='none')
        return eps_theta

    # Sampling step: x_{t-1} from x_t
    # @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, guidance_scale: float = 1.0, guidance_mode: Optional[str] = None, t_2: Optional[torch.Tensor] = None, training: bool = False) -> torch.Tensor:
        
        if training:
            self.model.train()
        else:
            self.model.eval()    
        
        betas_t = self.betas[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]

        # predict noise using the model
        if guidance_mode == 'none' or guidance_scale == 0.0:
            # Classifier-Free Guidance
            eps_theta = self.model(x_t, t, None if y is None else y)
        elif guidance_mode == 'cfg':
            eps_uncond = self.model(x_t, t, None)
            eps_cond   = self.model(x_t, t, y)
            w = guidance_scale
            eps_theta = (1 + w) * eps_cond - w * eps_uncond
            
        elif guidance_mode == 'autog':
            # Auto-Guidance
            assert self.bad_model is not None, "Bad model must be set for Auto-Guidance"
            eps_pos = self.model(x_t, t_2, None)
            eps_bad = self.bad_model(x_t, t, None)
            w = guidance_scale
            eps_theta = eps_pos + w * (eps_pos - eps_bad)
        else:
            raise ValueError(f"Unknown guidance mode: {guidance_mode}")

        # DDPM mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * eps_theta / sqrt_one_minus_alphas_bar_t)

        if training:
            return model_mean, eps_theta
        
        if (t == 0).all():
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            var = self.posterior_variance[t][:, None, None, None]
            return model_mean + torch.sqrt(var) * noise

    # Full sampling loop from pure noise
    @torch.no_grad()
    def sample(self, batch_size: int, labels: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None, guidance_mode: Optional[str] = None) -> torch.Tensor:
        self.model.eval()
        g = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        img = torch.randn(batch_size, self.cfg.channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
        
        if self.bad_model is not None:
            self.bad_model.eval()
        
        if labels is not None:
            labels = labels.to(self.device)
        
        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            # if t[0] == 0:
            img = self.p_sample(img, t, y=labels, guidance_scale=g, guidance_mode='none')  # no guidance at last step
            # else: 
            #     t_2 = torch.full((batch_size,), i // 2, device=self.device, dtype=torch.long)  # for time-step interpolation in Auto-Guidance
            #     img = self.p_sample(img, t, y=labels, guidance_scale=g, guidance_mode=guidance_mode, t_2=t_2)
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
                self.shadow[name] = p.data.clone().to(p.device)

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * p.data + self.decay * self.shadow[name].to(p.device)
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


class ConvQNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(base_channels * 8 * 8, 1)  

    def forward(self, x):
        # print(x.shape)
        h = self.conv(x)
        # print(h.shape)
        h = h.view(h.size(0), -1)
        # print(h.shape)
        score = self.fc(h)
        return score  # (B, 1)

# ----------------------------- Training Loop -------------------------------

@torch.no_grad()
def save_samples(ddpm: DDPM, out_dir: str, step: int, n: int = 16, guidance_scale: Optional[float] = None, guidance_mode: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    samples = ddpm.sample(n, guidance_scale=guidance_scale, guidance_mode=guidance_mode)
    # The training data is normalized to [-1, 1]; save_image can auto-normalize
    save_path = os.path.join(out_dir, f"samples_step_{step:07d}.png")
    save_image(samples, save_path, nrow=int(math.sqrt(n)), normalize=True, value_range=(-1, 1))
    logging.info(f"Saved samples to {save_path}")


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

    if cfg.guidance_mode == "autog":
        q_net = ConvQNet(in_channels=cfg.channels).to(device)
        q_opt = torch.optim.Adam(q_net.parameters(), lr=1e-4)
        logging.info("Initialized q_net for Auto-Guidance.")
    else:
        q_net, q_opt = None, None
    
    # loss, fid tracking
    fid_values_with_steps = []
    loss_values = []
    steps = []
    
    # bad model for Auto-Guidance
    resume_epochs = 0
    
    if cfg.guidance_mode == 'autog':
        assert cfg.ckpt_path is not None, "Please provide a checkpoint path for bad model in Auto-Guidance mode."
        ddpm.model = load_checkpoint(cfg.ckpt_path, model, ema, opt)[0].to(device)
        ddpm.get_bad_model_from_snapshot(cfg.bad_model_ckpt)
        logging.info("Bad model for Auto-Guidance is set.")

    if cfg.ckpt_path is not None:
        model, ema, opt, start_step = load_checkpoint(cfg.ckpt_path, model, ema, opt)
        logging.info(f"Resumed training from checkpoint: {cfg.ckpt_path} at step {start_step}")
        resume_epochs += int(cfg.ckpt_path.split('_')[-1].split('.')[0])  # continue for more epochs

    global_step = 0
    model.train()

    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}, {global_step}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device) if y is not None else None
            
            b = x.size(0)
            t = torch.randint(1, cfg.timesteps, (b,), device=device, dtype=torch.long)
            t_2 = torch.full((b,), t[0] // 2, device=device, dtype=torch.long)  # for time-step interpolation in Auto-Guidance
            use_uncond = torch.rand(1, device=x.device).item() < cfg.drop_cond_prob
            y_in = None if use_uncond else y

            # TODO
            # forward
            noise = torch.randn_like(x).to(device)
            x_t = ddpm.q_sample(x, t, noise)
            x_t_minus1_true = ddpm.q_posterior_mean(x, x_t, t)
            
            alpha = 0.1
            
            # model prediction
            model_mean, x_t_minus1_model = ddpm.p_sample(x_t, t, y_in, guidance_scale=0.0, guidance_mode='none', t_2=t_2, training=True)
                
            reg_weight = 0.6

            reg_loss = - F.mse_loss(x_t_minus1_model, x_t_minus1_true, reduction='mean')

            if cfg.guidance_mode == 'autog':
                x_t_minus1_model.requires_grad_(True)
                guidance_score = q_net(x_t_minus1_model)
                guidance_grad = torch.autograd.grad(
                    outputs=guidance_score.sum(),
                    inputs=x_t_minus1_model,
                    create_graph=False,
                    retain_graph=False,
                    # allow_unused=True
                )[0]
                x_t_minus1_guided = x_t_minus1_model + alpha * guidance_grad
                diffusion_loss = F.mse_loss(x_t_minus1_guided, x_t_minus1_true)
                
                # q_net loss
                score_real = q_net(x_t_minus1_true.detach())
                score_fake = q_net(x_t_minus1_model.detach())
                d_loss_real = F.mse_loss(score_real, torch.ones_like(score_real))
                d_loss_fake = F.mse_loss(score_fake, torch.zeros_like(score_fake))
                q_loss = 0.5 * (d_loss_real + d_loss_fake)
            else:
                diffusion_loss = reg_loss
                q_loss = torch.tensor(0.0, device=device)
            
            loss = diffusion_loss + q_loss
            loss.backward()
            
            # var_t = ddpm.posterior_variance[t][:, None, None, None]
            # logp_per_sample = -0.5 * (((x_t_minus1_true - model_mean) ** 2) / (var_t + 1e-8)).view(b, -1).sum(dim=1)
            # logp_mean = logp_per_sample.mean()
            # eps_theta = ddpm.p_sample(x_t, t-1, y_in, guidance_scale=0.0, guidance_mode=cfg.guidance_mode, t_2=t_2)
                
            # lambda_guidance = 1 - reg_weight
            
            # loss = reg_weight * reg_loss + lambda_guidance * eps_theta.mean()
            
            # loss.backward()
            # loss = -logp_mean - lambda_guidance * guidance_score
            
            with torch.no_grad():
                eps_pos = model(x_t, t_2, None).detach()
                eps_bad = ddpm.bad_model(x_t, t-1, None).detach()
                diff = (eps_pos - eps_bad).view(eps_pos.size(0), -1)
                l2_per_sample = (diff ** 2).sum(dim=1).cpu().numpy()
            
            # l2_per_sample = (diff ** 2).sum(dim=1).cpu()
            logging.info(f"Auto-Guidance L2 norm per sample: {l2_per_sample.mean():.4f} ± {l2_per_sample.std():.4f}")
            
            # loss.backward()
            
            # x_t_1 = ddpm.p_sample(x, t, y_in, guidance_scale=cfg.guidance_scale, guidance_mode='none', t_2=t_2, training=True)
            # eps_theta = ddpm.p_losses(x_t_minus1_model, t, y_in, guidance_scale=cfg.guidance_scale, train_with_autog=cfg.train_with_autog, t_2=t_2)
            
            
            # loss = ((x_t_minus1_true - x_t_minus1_model) ** 2).mean()
            
            # torch.autograd.backward(
            #     tensors=x_t_minus1_model,
            #     grad_tensors=eps_theta,
            #     retain_graph=True
            # )
            
            # loss.backward()
                        
            # print(eps_theta.shape, x_t_1.shape)
            
            # grads = torch.autograd.grad(
            #     outputs=x_t_1,
            #     inputs=model.parameters(),
            #     grad_outputs=eps_theta,
            #     create_graph=False,
            #     retain_graph=False,
            #     only_inputs=True,
            #     allow_unused=True
            # )
            
            # # alpha = 0.1

            
            # for p, g in zip(model.parameters(), grads):
            #     if g is not None:
            #         # if p.grad is None:
            #         p.grad = g
                    
            # print(grads[0])
            # loss = loss.sum()
            # loss.backward()
            loss_values.append(loss.item())
            logging.info(f"Step {global_step}, Loss: {loss.item():.4f}")
            # loss_values.append(0)
            if (global_step + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)

            
            
            # if global_step % 500 == 0:
            #     logging.info(f"epoch {epoch} step {global_step} loss ")

            if global_step % 5000 == 0 and global_step > 0:
                # Save samples with EMA weights for better quality
                ema.apply_shadow(model)
                save_samples(ddpm, os.path.join(cfg.logdir, "samples"), step=global_step, n=16, guidance_scale=cfg.guidance_scale, guidance_mode=cfg.guidance_mode)
                fid = compute_fid(ddpm, train_loader, device, out_dir=os.path.join(cfg.logdir, "fid"), n_samples=5000, batch_size=128 , guidance_scale=cfg.guidance_scale, guidance_mode=cfg.guidance_mode, logdir=cfg.logdir, step=global_step)
                fid_values_with_steps.append((fid, global_step, ddpm.model.state_dict()))
                steps.append(global_step)
                current_fid_difference, idx = 0, len(fid_values_with_steps)-1
                
                # update bad model if fid improves by more than threshold
                while current_fid_difference > cfg.fid_threshold and len(fid_values_with_steps) > 1 and idx > 0:
                    current_fid, step, snapshot = fid_values_with_steps[idx]
                    previous_fid, _, _ = fid_values_with_steps[idx-1]
                    current_fid_difference = previous_fid - current_fid
                    if current_fid_difference > cfg.fid_threshold:
                        logging.info(f"Updating bad model from step {step} with FID improvement {current_fid_difference:.4f}")
                        ddpm.update_bad_model(snapshot)
                    idx -= 1
                
                ema.restore(model)

            global_step += 1

        # checkpoint per epoch
        if (epoch + 1) % cfg.ckpt_every == 0:
            os.makedirs(cfg.logdir, exist_ok=True)
            ckpt_path = os.path.join(cfg.logdir, f"ckpt_epoch_{resume_epochs+epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow,
                "opt": opt.state_dict(),
                "cfg": cfg.__dict__,
                "step": global_step,
            }, ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")

    only_fid_values = [f[0] for f in fid_values_with_steps]

    plot_all(loss_values, only_fid_values, out_dir=cfg.logdir, mode=cfg.guidance_mode, steps=steps)

def plot_loss(loss_values, out_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_values)
    plt.title('Training Loss over Time')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    for i, v in enumerate(loss_values):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=6, rotation=45)
    
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved loss plot to {out_path}")

def plot_all(loss_values, fid_values, out_dir, mode=None, steps=None):
    os.makedirs(out_dir, exist_ok=True)
    plot_loss(loss_values, os.path.join(out_dir, f"{mode}_training_loss.png"))
    plot_fid(fid_values, os.path.join(out_dir, f"{mode}_fid_over_time.png"), global_step=steps)

def load_checkpoint(ckpt_path: str, model: nn.Module, ema: EMA, opt: torch.optim.Optimizer):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    ema.shadow = checkpoint['ema']
    opt.load_state_dict(checkpoint['opt'])
    step = checkpoint.get('step', 0)
    logging.info(f"Loaded checkpoint from {ckpt_path} at step {step}")
    return model, ema, opt, step

# ----------------------------- Evaluation ---------------------------------
def eval_fid(cfg: DDPMConfig):
    device = default_device()
    _, test_loader = get_dataloaders(data_dir=os.path.join(cfg.logdir, "data"), batch_size=cfg.batch_size)

    model = UNet(in_ch=cfg.channels, base_ch=128, ch_mults=(1, 2, 2, 2), time_emb_dim=512, with_attn=(False, True, True, False), num_classes=cfg.num_classes)
    model.to(device)

    ddpm = DDPM(model, cfg, device)

    if cfg.ckpt_path is not None:
        model, _, _, _ = load_checkpoint(cfg.ckpt_path, model, EMA(model), torch.optim.AdamW(model.parameters(), lr=cfg.lr))
        logging.info(f"Loaded model from checkpoint: {cfg.ckpt_path}")

    fid = compute_fid(ddpm, test_loader, device, out_dir=os.path.join(cfg.logdir, "fid_eval"), n_samples=5000, batch_size=128 , guidance_scale=cfg.guidance_scale, guidance_mode=cfg.guidance_mode)
    logging.info(f"Final FID on test set: {fid:.4f}")

# ----------------------------- FID Computation -----------------------------
@torch.no_grad()
def compute_fid(ddpm: DDPM, data_loader, device: torch.device, out_dir: str, n_samples: int = 5000, batch_size: int = 128, guidance_scale: Optional[float] = None, guidance_mode: Optional[str] = None, logdir: Optional[str] = None, step: Optional[int] = None) -> float:
    """
    Compute FID score between generated samples and real CIFAR-10 images.

    Args:
        ddpm: Trained DDPM instance
        data_loader: real CIFAR-10 dataloader (for true samples)
        device: torch device
        out_dir: directory to save temporary samples
        n_samples: number of generated samples to compare
        batch_size: batch size for sampling
    Returns:
        fid (float)
    """
    os.makedirs(out_dir, exist_ok=True)
    gen_dir = os.path.join(out_dir, "generated")
    real_dir = os.path.join(out_dir, "real")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    # --- Generate fake images ---
    total = 0
    while total < n_samples:
        cur_bs = min(batch_size, n_samples - total)
        samples = ddpm.sample(cur_bs, guidance_scale=guidance_scale, guidance_mode=guidance_mode).detach().cpu()
        for i in range(cur_bs):
            save_path = os.path.join(gen_dir, f"{total+i:06d}.png")
            save_image(samples[i], save_path, normalize=True, value_range=(-1, 1))
        total += cur_bs

    # --- Save real images (same n_samples for fair FID) ---
    total = 0
    for x, _ in data_loader:
        for i in range(x.size(0)):
            if total >= n_samples:
                break
            save_path = os.path.join(real_dir, f"{total:06d}.png")
            save_image(x[i], save_path, normalize=True, value_range=(-1, 1))
            total += 1
        if total >= n_samples:
            break

    # --- Compute FID ---
    paths = [real_dir, gen_dir]
    fid_value = fid_score.calculate_fid_given_paths(paths, batch_size, device, dims=2048)
    logging.critical(f"FID: {fid_value:.4f}")

    # --- Logging ---
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        log_path = os.path.join(logdir, "fid_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{step if step is not None else 'NA'}\t{fid_value:.6f}\n")
        logging.info(f"Logged FID to {log_path}")

    return fid_value

def plot_fid(fid_values, out_path, global_step=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(global_step, fid_values, marker='o')
    plt.title('FID over Time')
    plt.xlabel('Evaluation Step')
    plt.ylabel('FID')
    plt.grid(True)
    
    for i, v in enumerate(fid_values):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=6, rotation=45)
        
    plt.savefig(out_path)
    plt.close()
    print(f"Saved FID plot to {out_path}")

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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--logdir", type=str, default="runs/ddpm")
    p.add_argument("--ckpt_every", type=int, default=50)
    
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--drop_cond_prob", type=float, default=0.1)
    p.add_argument("--guidance_mode", type=str, default='cfg', choices=['cfg', 'autog', 'none'])
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--train_with_autog", type=str, help="Whether to use Auto-Guidance during training (requires bad model)", default='False')
    
    # p.add_argument("--device", type=str, default='cuda', help="Device to use (default: cuda if available)")
    p.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to bad model")
    p.add_argument("--bad_model_ckpt", type=str, default='/home/jasonx62301/for_python/Auto-Guidance-test/runs/ddpm/ckpt_epoch_200.pt', help="Path to bad model checkpoint for Auto-Guidance")
    return p


def main():
    cfg = DDPMConfig(**vars(build_argparser().parse_args()))
    setup_logging(cfg.logdir)
    train(cfg)


if __name__ == "__main__":
    main()
