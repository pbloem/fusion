import fusion
from fusion import tile, batch
from fusion.tools import here, d, fc, gradient_norm, tic, toc, prod, data

import fire, math
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dst

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from math import ceil

from pathlib import Path

import tqdm
from tqdm import trange

import importlib.util
import sys

import wandb

def train(
        epochs=5,
        steps=120,
        lr=3e-4,
        bs=16,
        limit=float('inf'), # limits the number of batches per epoch,
        data_name='mnist',
        data_dir='./data',
        size=(32, 32),
        num_workers=2,
        grayscale=False,
        dp=False,
        unet_channels=(16, 32, 64),  # Basic structure of the UNet in channels per block
        blocks_per_level=3,
        sample_bs=16,
        plot_every=1,
        eval_steps=20,
        dres=8,             # resolution of the tiles in the degradation
        beta=1.0,
        name='vcd',
        debug=False,
        time_emb=512,
        sample_mix = 1.0, # how much of the batch to augment
        p = 1.0, # exponent for sampling the time offsets. >1.0 makes time values near the target value more likely
):

    """
    Variational cold diffusion
    """

    if wandb is not None:
        wd = wandb.init(
            name = name,
            project = 'vcd',
            tags = [],
            config =locals(),
            mode = 'disabled' if debug else 'online'
        )

    h, w = size

    scaler = torch.cuda.amp.GradScaler()

    tic()
    dataloader, (h, w), n = data(data_name, data_dir, batch_size=bs, nw=num_workers, grayscale=grayscale, size=size)
    print(f'data loaded ({toc():.4} s)')

    unet = fusion.VCUNet(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level,
                         mid_layers=3, time_emb=time_emb)

    if torch.cuda.is_available():
        unet = unet.cuda()

    if dp:
        unet = torch.nn.DataParallel(unet)
    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())

    Path('./samples_vcd/').mkdir(parents=True, exist_ok=True)

    for e in range(epochs):
        # Train
        unet.train()
        for i, (btch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            if torch.cuda.is_available():
                btch = btch.cuda()

            b, c, h, w = btch.size()

            # Corruption times

            # -- pick three random points in (0, 1) and sort them
            # t = torch.rand(size=(b, 3), device=d())
            # torch.sort(t, dim=1)[0]
            # this is biased

            with torch.no_grad():
                # pick t0 uniform over (0, 1)
                t = torch.rand(size=(b, 3), device=d())
                t[:, 1:] **= p # adjust to make nearby points more likely

                # t1 is uniform over the range between t0 and 1
                t[:, 1] = t[:, 1] * (1 - t[:, 0]) + t[:, 0]

                #t2 is uniform over the range between t1 and 1
                t[:, 2] = t[:, 2] * (1 - t[:, 1]) + t[:, 1]

            xs = [batch(btch, op=tile, t=t[:, i], nh=dres, nw=dres) for i in range(3)]

            # Sample one step to augment the data (t2 -> t1)
            with torch.no_grad():
                diff = unet(x1=xs[2], x0=None, t1=t[:, 2], t0=t[:, 1]) #.sigmoid()
                x1p = xs[2] + diff

                sel = torch.rand(size=(b,), device=d()) < sample_mix
                # idx = (~ sel).nonzero()
                x1p[~ sel] = xs[1][~ sel] # reset a proportion to the non-augmented batch

            # Predict x0 from x1p (t1 -> t0)
            output, kls = unet(x1=x1p, x0=xs[0], t1=t[:, 1], t0=t[:, 0])
            # output = output.sigmoid()

            diff = xs[0] - xs[1] # predict the delta between x0 and x1
            rc_loss = ((output - diff) ** 2.0).reshape(b, -1).sum(dim=1) # Simple loss

            loss = (rc_loss + beta * sum(kls)).mean()

            loss.backward()
            opt.step()

            if wandb:
                wandb.log({
                    'loss': loss.item(),
                    'kl_loss': sum(kls).mean().item(),
                    'gradient_norm': gradient_norm(unet),
                })

            bar.set_postfix({'loss' : loss.item()})
            opt.zero_grad()


        # # Sample
        print('Generating sample, epoch', e)
        unet.eval()
        with torch.no_grad():

            ims = torch.randn(size=(sample_bs, c, h, w), device=d())
            ims = batch(ims, op=tile, t=1.0, nh=dres, nw=dres)

            delta = 1.0 / eval_steps
            n = 0
            for t in (1 - torch.arange(delta, 1, 1/20)):

                texp = t.expand((ims.size(0),)).to(d())
                diff = unet(x1=ims, x0=None, t0=texp-delta, t1=texp) #.sigmoid()
                ims = ims + diff

                n += 1
                if n % plot_every == 0:
                    grid = make_grid(ims.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.gca().axis('off')
                    plt.savefig(f'./samples_vcd/denoised-{e}-{n:05}.png')

                    # grid = make_grid(mutilde.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                    # plt.imshow(grid)
                    # plt.gca().axis('off')
                    # plt.savefig(f'./samples_vcd/mean-{e}-{t:05}.png')


if __name__ == '__main__':
    fire.Fire()