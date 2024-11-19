import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch
import matplotlib.pyplot as plt
from scripts.unet import SinusoidalPositionEmbeddings
from contextlib import contextmanager
from copy import deepcopy
import math

from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm,trange
from einops import rearrange, repeat, reduce, pack, unpack
from torch import nn, einsum

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


timesteps = 1000

# compute betas
betas = linear_beta_schedule(timesteps=timesteps)

# compute alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# calculations for the forward diffusion q(x_t | x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


# utility function to extract the appropriate t index for a batch of indices.
# e.g., t=[10,11], x_shape=[b,c,h,w] --> a.shape = [2,1,1,1]
# e.g., t=[7,12,15,20], x_shape=[b,h,w] --> a.shape = [4,1,1]
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)  # z (it does not depend on t!)

    # adjust the shape
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# def get_noisy_image(x_start, t):
#     x_noisy = q_sample(x_start, t=t)  # add noise
#     noisy_image = reverse_transform(x_noisy.squeeze())  # turn back into PIL image
#
#     return noisy_image

# Let's look at how the time embeddings look like


time_emb = SinusoidalPositionEmbeddings(100)
t1 = time_emb(torch.tensor([10]))
t2 = time_emb(torch.tensor([10.2]))
t3 = time_emb(torch.tensor([-10]))


def l2norm(t):
    return F.normalize(t, dim = -1)
def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t
class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)
# Define the model (a residual U-Net)
def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x
class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_out, 3, padding=1),
            nn.ReLU(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None].repeat([1, 1, shape[2]])




class Diffusion(nn.Module):
    def __init__(self, inchannel =1,outchannel=1):
        super().__init__()
        c = 32  # The base channel count

        #[8, 32, 64, 128, 256, 512, 512, 1024, 1024]
        #[32,64,128]
        #factors=[1, 4, 4, 4, 2, 2, 2, 2, 2]
        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        #self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(   # 1
            ResConvBlock(inchannel + 16, c, c), #32
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool1d(4),  # 1-> 4
                ResConvBlock(c, c * 2, c * 2), #64
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool1d(4),  # 4-> 16
                    ResConvBlock(c * 2, c * 4, c * 4), #128
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool1d(4),  # 16 -> 64
                        ResConvBlock(c * 4, c * 8, c * 8), #256
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SkipBlock([
                            nn.AvgPool1d(2),  # 64 -> 128
                            ResConvBlock(c * 8, c * 16, c * 16), #512
                            ResConvBlock(c * 16, c * 16, c * 16),
                            SkipBlock([
                                nn.AvgPool1d(2),  # 128 -> 256
                                ResConvBlock(c * 16, c * 16, c * 16), #512
                                ResConvBlock(c * 16, c * 16, c * 16),
                                SkipBlock([
                                    nn.AvgPool1d(2),  # 256 -> 512
                                    ResConvBlock(c * 16, c * 32, c * 32), #1024
                                    ResConvBlock(c * 32, c * 32, c * 32),
                                    SkipBlock([
                                        nn.AvgPool1d(2),  # 512 -> 1024
                                        ResConvBlock(c * 32, c * 32, c * 32), #1024
                                        ResConvBlock(c * 32, c * 32, c * 32),
                                        ResConvBlock(c * 32, c * 32, c * 32), #1024
                                        ResConvBlock(c * 32, c * 32, c * 32),
                                        nn.Upsample(scale_factor=2),
                                    ]),
                                    ResConvBlock(c * 64, c * 32, c * 32), #1024
                                    ResConvBlock(c * 32, c * 32, c * 16),
                                    nn.Upsample(scale_factor=2),
                                ]),
                                ResConvBlock(c * 32, c * 16, c * 16), #1024
                                ResConvBlock(c * 16, c * 16, c * 16),
                                nn.Upsample(scale_factor=2),
                            ]),
                            ResConvBlock(c * 32, c * 16, c * 16), #1024
                            ResConvBlock(c * 16, c * 16, c * 8),
                            nn.Upsample(scale_factor=2),
                        ]),
                        ResConvBlock(c * 16, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=4),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=4),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=4),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, outchannel),
        )

    def forward(self, input, log_snrs):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
        #class_embed = expand_to_planes(self.class_embed(cond), input.shape)
        return self.net(torch.cat([input,  timestep_embed], dim=1))


def p_losses(denoise_model, x_start, t):
    # random sample z
    noise = torch.randn_like(x_start)

    # compute x_t
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # recover z from x_t with the NN
    predicted_noise = denoise_model(x_noisy, t)

    loss = torch.mean((noise - predicted_noise) ** 2)

    return loss


# calculations for posterior q(x_{t-1} | x_t, x_0) = q(x_{t-1} | t, x_0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # β_t


@torch.no_grad()
def p_sample(model, x, t, t_index):
    # adjust shapes
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Use the NN to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    # Draw the next sample
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)  # beta_t
        noise = torch.randn_like(x)  # z
        return model_mean + torch.sqrt(posterior_variance_t) * noise  # x_{t-1}


# Sampling loop
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    print(img.shape)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((shape[0],), i, device=device, dtype=torch.long), i)
        imgs.append(img)
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=1):
    return p_sample_loop(model, shape=(batch_size, channels, image_size))