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
        sched = 'uniform',
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

    # degradation color
    fv = torch.tensor((.3, .9, .0))[None, :, None, None]

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

            if sched == 'uniform':
                with torch.no_grad():
                    # pick t0 uniform over (0, 1)
                    t = torch.rand(size=(b, 3), device=d())
                    t[:, 1:] **= p # adjust to make nearby points more likely`

                    # t1 is uniform over the range between t0 and 1
                    t[:, 1] = t[:, 1] * (1 - t[:, 0]) + t[:, 0]

                    #t2 is uniform over the range between t1 and 1
                    t[:, 2] = t[:, 2] * (1 - t[:, 1]) + t[:, 1]

            elif sched == 'discrete':
                with torch.no_grad():
                    max = dres ** 2
                    t = torch.randint(low=2, high=max + 1, size=(b, 1))
                    t = torch.cat([t, t-1, t-2], dim=1).to(torch.float)
                    t = t / max
            else:
                fc(sched, 'sched')

            xs = [batch(btch, op=tile, t=t[:, i], nh=dres, nw=dres, fv=fv) for i in range(3)]

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

            diff = xs[0] - x1p # predict the delta between x0 and x1p

            rc_loss = ((output - diff) ** 2.0).reshape(b, -1).sum(dim=1) # Simple loss
            kls = sum(kl.reshape(b, -1).sum(dim=-1) for kl in kls)

            loss = (rc_loss + beta * kls).mean()

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

            btch = btch[torch.randperm(btch.size(0))]

            # plot an illustration of the sampling process
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 6))

            # max = dres ** 2
            # p = max//2
            # ts = (p-1)/max, p/max, (p+1)/max

            max = dres ** 2
            ts = (max-2)/max, (max-1)/max, 1.0

            ts = [torch.tensor(t, device=d()).expand((btch.size(0),)) for t in ts]
            xs = [batch(btch.to(d()), op=tile, t=t, nh=dres, nw=dres, fv=fv) for t in ts]

            plotim(xs[0][0], axs[0][0]); axs[0][0].set_title('x0')
            plotim(xs[1][0], axs[0][1]); axs[0][1].set_title('x1')
            plotim(xs[2][0], axs[0][2]); axs[0][2].set_title('x2')

            for i in range(3):
                diff = unet(x1=xs[2], x0=None, t1=ts[2], t0=ts[1])  # .sigmoid()
                x1p = xs[2] + diff

                plotim(x1p[0], axs[1][i]); axs[1][0].set_ylabel('x1 augmented')

                output, kls = unet(x1=x1p, x0=xs[0], t1=ts[1], t0=ts[0])
                pred = x1p + output

                plotim(pred[0], axs[2][i]); axs[2][0].set_ylabel('x0 pred')

            plt.savefig(f'./samples_vcd/snapshot-{e}.png')

            plt.figure(figsize=(4, 4))

            # plot a bunch of samples
            ims = torch.randn(size=(sample_bs, c, h, w), device=d())
            ims = batch(ims, op=tile, t=1.0, nh=dres, nw=dres, fv=fv)

            steps = dres**2
            delta = 1.0 / steps
            n = 0
            for t in (1 - torch.arange(delta, 1, 1/steps)):

                texp = t.expand((ims.size(0),)).to(d())
                diff = unet(x1=ims, x0=None, t0=texp-delta, t1=texp) #.sigmoid()
                ims = ims + diff

                n += 1
                griddle(ims, f'./samples_vcd/denoised-{e}-{n:05}.png')

def plotim(im, ax):
    ax.imshow(im.permute(1, 2, 0).cpu().clip(0, 1))
    ax.axis('off')

def griddle(btch, file):
    grid = make_grid(btch.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
    plt.imshow(grid)
    plt.gca().axis('off')
    plt.savefig(file)

if __name__ == '__main__':
    fire.Fire()