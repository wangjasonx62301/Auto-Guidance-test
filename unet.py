import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        scale = head_dim ** -0.5
        attn = torch.softmax(torch.einsum('bncd,bncd->bnc', q * scale, k), dim=-1)
        out = torch.einsum('bnc,bncd->bncd', attn, v)
        out = out.reshape(b, c, h, w)
        return x_in + self.proj(out)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, with_attn=False):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch) if with_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, with_attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResidualBlock(out_ch * 2, out_ch, time_emb_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch) if with_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, ch_mults=(1, 2, 2, 2), time_emb_dim=512, with_attn=(False, True, True, False)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        downs = []
        ch = base_ch
        self.skips_channels = []
        for i, m in enumerate(ch_mults):
            out_ch = base_ch * m
            downs.append(Down(ch, out_ch, time_emb_dim, with_attn=with_attn[i]))
            self.skips_channels.append(out_ch)
            ch = out_ch
        self.downs = nn.ModuleList(downs)

        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim)

        ups = []
        for i, m in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * m
            ups.append(Up(ch, out_ch, time_emb_dim, with_attn=with_attn[i]))
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.final_norm = nn.GroupNorm(8, ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

        self.time_emb_dim = time_emb_dim

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.time_emb_dim // 2)
        t_emb = self.time_mlp(t_emb)
        x = self.init_conv(x)
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb)
        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x