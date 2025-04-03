import fusion
from fusion import tile, batch
from fusion.tools import here, d, fc, gradient_norm, tic, toc, prod, data, sample, kl_loss

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

def source(n):
    """
    Samples N points from a mixture of two Gaussians
    """

    x = torch.randn(n)
    x[:n//2] += 3
    x[n//2 :] -= 3

    return x

def siginv(p):
  return torch.log(p/(1-p))

def quantize(x, t):

  level = ((MAXQ-MINQ) * (1 - t)).floor() + MINQ

  x = torch.sigmoid(x)
  chunk = 2 ** (-level)
  u = x > 0.5
  x[u]  = (x[u]/chunk[u]).floor() * chunk[u]
  x[~u] = (x[~u]/chunk[~u]).ceil() * chunk[~u]

  return siginv(x)

MINQ, MAXQ = 1, 12
EPS = 1e-8

def train(
        epochs=5,
        steps=120,
        lr=3e-4,
        bs=16,
        num_batches=50000,
        hidden=64,
        num_workers=2,
        grayscale=False,
        dp=False,
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

    N = 500
    T = 128
    delta = 1 / T

    #Plot source
    plt.hist(source(10_000), bins=100)
    plt.savefig('hist.png')
    plt.figure()

    if wandb is not None:
        wd = wandb.init(
            name = name,
            project = 'vcd-1d',
            tags = [],
            config =locals(),
            mode = 'disabled' if debug else 'online'
        )

    net = nn.Sequential( # Denoiser
        nn.Linear(4, hidden), # input is [x, t_from, t_to, z], 1d each, t_from is closer to 1 than t_to
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 2) # output is mu and sigma on x
    )

    enc = nn.Sequential( # Encoder
        nn.Linear(2, hidden), # input is [x, t_from], 1d each
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 2) # output is mu and sigma on z
    )

    if torch.cuda.is_available():
        print('Cuda found.')
        net = net.cuda()
        enc = enc.cuda()

    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=net.parameters())

    for i in (bar := trange(num_batches)):

        btch = source(bs)

        if torch.cuda.is_available():
            btch = batch.cuda()

        # Corruption times
        if sched == 'uniform':
            with torch.no_grad():
                # pick t0 uniform over (0, 1)
                # t = torch.rand(size=(bs, 3), device=d())
                t = torch.rand(size=(bs, 3), device=d()) * (1-2*delta)
                t[:, 1:] **= p # adjust to make nearby points more likely`
                # t[:, 1] = t[:, 1] * (1 - t[:, 0]) + t[:, 0] # t1 is over the range between t0 and 1
                t[:, 1] = t[:, 0] - delta
                # t[:, 2] = t[:, 2] * (1 - t[:, 1]) + t[:, 1] # t2 is over the range between t1 and 1
                t[:, 2] = t[:, 0] - 2 * delta

        # elif sched == 'discrete':
        #     with torch.no_grad():
        #         max = dres ** 2
        #         t = torch.randint(low=2, high=max + 1, size=(b, 1))
        #         t = torch.cat([t, t-1, t-2], dim=1).to(torch.float)
        #         t = t / max

        else:
            fc(sched, 'sched')

        # print(t)
        # print(t.size())
        # print(btch.size())

        xs = [quantize(btch, t=t[:, i]) for i in range(3)]

        # Sample one step to augment the data (t2 -> t1)
        with torch.no_grad():
            z = torch.randn(bs)

            out = net(torch.cat([xs[2][:, None], t[:, 2][:, None], t[:, 1][:, None], z[:, None]], dim=1))
            norm = dst.Normal(out[:, 0], out[:, 1].exp() + EPS)
            x1p = norm.sample()

            sel = torch.rand(size=(bs,), device=d()) < sample_mix
            # idx = (~ sel).nonzero()
            x1p[~ sel] = xs[1][~ sel] # reset a proportion to the non-augmented batch

        # Predict x0 from x1p (t1 -> t0)
        # output, kls = net(x1=x1p, x0=xs[0], t1=t[:, 1], t0=t[:, 0])
        # output = output.sigmoid()

        zms = enc(torch.cat([x1p[:, None], t[:, 1][:, None]], dim=1))
        z = sample(zms[:, 0], zms[:, 1])

        out = net(torch.cat([x1p[:, None], t[:, 1][:, None], t[:, 0][:, None], z[:, None]], dim=1))
        norm = dst.Normal(out[:, 0], out[:, 1].exp() + EPS)

        rec_loss = - norm.log_prob(xs[0])
        kl = kl_loss(zms[:, 0], zms[:, 1])
        loss = (rec_loss + beta * kl).mean()

        loss.backward()
        opt.step()

        if wandb:
            wandb.log({
                'loss': loss.item(),
                'kl_loss': kl.mean().item(),
                'gradient_norm': gradient_norm(net),
            })

        bar.set_postfix({'loss' : loss.item(), 'kl' : kl.mean().item(), 'ms': out[:, 1].exp().mean().item()})
        opt.zero_grad()

    # # Sample
    print('Training done, evaluating')
    with torch.no_grad():


        x = torch.zeros(N)

        plt.scatter(torch.ones(N), x, c='b', alpha=0.1, s=5, linewidth=0)

        for t in torch.linspace(1, 0, T):

            zs = torch.randn(N)
            out = net(torch.cat([x[:, None], t.expand(N, 1), (t-delta).expand(N, 1), zs[:, None]], dim=1))
            norm = dst.Normal(out[:, 0], out[:, 1].exp() + EPS)
            x = norm.sample()
            # print(x[:5])
            # res.append(x.clone())

            plt.scatter((t-delta).expand(N), x, c='b', alpha=0.1, s=5, linewidth=0)

        plt.savefig('refinement.png')

        plt.figure()

        # Plot degradation
        x = source(N)

        plt.scatter(torch.zeros(N), x, c='b', alpha=0.1, s=12, linewidth=0)

        for t in torch.linspace(0, 1, T):
            plt.scatter((t+delta).expand(N), quantize(x, (t+delta).expand(N)), c='r', alpha=0.1, s=5, linewidth=0)

        plt.savefig('degradation.png')


if __name__ == '__main__':
    fire.Fire()