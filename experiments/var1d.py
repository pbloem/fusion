import fusion
from fusion import tile, batch
from fusion.tools import here, d, fc, gradient_norm, tic, toc, prod, data, sample, kl_loss, gsample, gkl, nkl

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


def intsource(num, n=2):
    return (torch.rand(num) * n).floor()[:, None]

def source(n):
    """
    Samples N points from a mixture of two Gaussians
    """

    x = torch.randn(n)
    x[:n//2] += 3
    x[n//2 :] -= 3

    return x[:, None]

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

def lquantize(x, level):

    if type(level) is not torch.Tensor:
      level = torch.tensor([level])
      level = level.expand(*x.size())

    x = torch.sigmoid(x)
    chunk = 2.0 ** (-level)
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
        nn.Linear(hidden, 2), # output is mu and sigma on x
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

    opt = torch.optim.Adam(lr=lr, params=list(net.parameters()) + list(enc.parameters()))

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

        xs = [quantize(btch, t=t[:, i:i+1]) for i in range(3)]

        # Sample one step to augment the data (t2 -> t1)
        with torch.no_grad():
            z = torch.randn(bs, 1)

            out = net(torch.cat([xs[2], t[:, 1:2], t[:, 1:2], z], dim=1))
            norm = dst.Normal(out[:, :1], out[:, 1:].exp() + EPS)
            x1p = norm.sample()

            sel = torch.rand(size=(bs,), device=d()) < sample_mix
            # idx = (~ sel).nonzero()
            x1p[~ sel] = xs[1][~ sel] # reset a proportion to the non-augmented batch

        # Predict x0 from x1p (t1 -> t0)
        # output, kls = net(x1=x1p, x0=xs[0], t1=t[:, 1], t0=t[:, 0])
        # output = output.sigmoid()

        zms = enc(torch.cat([x1p, t[:, 1:2]], dim=1))
        z = sample(zms[:, :1], zms[:, 1:])

        out = net(torch.cat([x1p, t[:, 1:2], t[:, 0:1], z], dim=1))
        norm = dst.Normal(out[:, :1], out[:, 1:].exp() + EPS)

        rec_loss = - norm.log_prob(xs[0])
        kl = kl_loss(zms[:, :1], zms[:, 1:])
        loss = rec_loss + beta * kl
        assert rec_loss.size() == kl.size() == loss.size()

        loss = loss.mean()
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


        x = torch.zeros(N, 1)

        plt.scatter(torch.ones(N), x, c='b', alpha=0.1, s=5, linewidth=0)

        for t in torch.linspace(1, 0, T):

            zs = torch.randn(N, 1)
            out = net(torch.cat([x, t.expand(N, 1), (t-delta).expand(N, 1), zs], dim=1))
            norm = dst.Normal(out[:, :1], out[:, 1:].exp() + EPS)
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
            plt.scatter((t+delta).expand(N), quantize(x, (t+delta).expand(N, 1)), c='r', alpha=0.1, s=5, linewidth=0)

        plt.savefig('degradation.png')

def isolated(
    level=1,
    hidden=16,
    num_batches=500,
    lr=3e-4,
    bs=128,
    beta=1.0,
    smult=1e-10,
    sadd=1,
    smax=None,
):
    """
    Isolates one step of the degradation/refinement and checks how well the model can learn that specific part.
    :param level:
    :return:
    """

    def sig(raw):
        s = (raw * smult + sadd).exp() + EPS
        if smax is None:
            return s
        return s.clip(0, smax)

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

    opt = torch.optim.Adam(lr=lr, params=list(net.parameters()) + list(enc.parameters()))

    for i in (bar := trange(num_batches)):

        x = source(bs)

        t = torch.zeros(bs, 1)

        x0 = lquantize(x, level=level+1) # less degraded (level 1 = fully degraded)
        x1 = lquantize(x, level=level) # more degraded

        zms = enc(torch.cat([x0, t], dim=1))
        z = sample(zms[:, :1], zms[:, 1:])

        out = net(torch.cat([x1, t, t, z], dim=1))
        norm = dst.Normal(out[:, :1], sig(out[:, 1:]))

        rec = - norm.log_prob(x0)
        kl = kl_loss(zms[:, :1], zms[:, 1:])
        loss = rec + beta * kl

        assert rec.size() == kl.size() == loss.size() == (bs, 1)
        loss = loss.mean()

        loss.backward()
        opt.step()

        bar.set_postfix({
            'loss' : loss.item(),
            'kl' : kl.mean().item(),
            'ms': sig(out[:, 1]).mean().item()})
        opt.zero_grad()

    # # Sample
    print('Training done, evaluating')
    with torch.no_grad():
        N = 5000
        x = source(N)

        t = torch.zeros(N, 1)

        x0 = lquantize(x, level=level+1) # less degraded
        x1 = lquantize(x, level=level)

        plt.subplot(1, 2, 1)

        plt.scatter(x1, x0, c='b', alpha=0.1, linewidth=0, s=5)
        plt.ylabel('x0 (less degraded)')
        plt.xlabel('x1 (more degraded)')
        plt.title('true')

        zms = enc(torch.cat([x0, t], dim=1))
        z = sample(zms[:, :1], zms[:, 1:])

        out = net(torch.cat([x1, t, t, z], dim=1))
        norm = dst.Normal(out[:, :1], sig(out[:, 1:]) )
        x0p = norm.sample()

        plt.subplot(1, 2, 2)

        plt.scatter(x1, x0p, c='r', alpha=0.1, linewidth=0, s=5)
        plt.ylabel('x0 (predicted)')
        plt.xlabel('x1 (more degraded)')
        plt.title('inverse')

        plt.tight_layout()
        plt.savefig('isolated_recon.png')

        plt.subplot(1, 2, 1)

        plt.scatter(x1, x0, c='b', alpha=0.1, linewidth=0, s=5)
        plt.ylabel('x0 (less degraded)')
        plt.xlabel('x1 (more degraded)')
        plt.title('true')

        z = torch.randn(N, 1)
        out = net(torch.cat([x1, t, t, z], dim=1))
        norm = dst.Normal(out[:, :1], sig(out[:, 1:]) )
        x0p = norm.sample()

        plt.subplot(1, 2, 2)

        plt.scatter(x1, x0p, c='r', alpha=0.1, linewidth=0, s=5)
        plt.ylabel('x0 (predicted)')
        plt.xlabel('x1 (more degraded)')
        plt.title('inverse')

        plt.tight_layout()
        plt.savefig('isolated_sample.png')

def vae(
    bs=128,
    n=2,
    hidden=64,
    lr=3e-4,
    num_batches=2_500,
    beta=1.0,
    skip_sample = False,
    generalized = False,
    prior_alpha = 2.0,
):
    """
    Very simple VAE experiment on n integer values.
    """

    enc = nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 3) # output is mu, log-sigma, log-alpha on z
    )

    dec = nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 2) # output is mu and sigma on x
    )

    opt = torch.optim.Adam(lr=lr, params=list(enc.parameters()) + list(dec.parameters()))

    for i in (bar := trange(num_batches)):

        x = intsource(num=bs, n=n)

        zms = enc(x)
        if skip_sample:
            z = zms[:, :1]
        else:
            if generalized:
                z = gsample(zms[:, 0:1], zms[:, 1:2].exp(), zms[:, 2:].exp() )

                if random.random() < 0.01:
                    print('alphas', zms[:, 2:].exp()[:10])
                    print('sigmas', zms[:, 1:2].exp()[:10])

            else:
                z = sample(zms[:, 0:1], zms[:, 1:2])

        assert z.size() == (bs, 1)

        out = dec(z)
        norm = dst.Normal(out[:, :1], out[:, 1:].exp() + EPS)

        rec_loss = - norm.log_prob(x)

        if generalized:
            kl = gkl(zms[:, :1], zms[:, 1:2].exp(), zms[:, 2:].exp(), alpha2=torch.tensor([prior_alpha]), samples=100)
        else:
            kl = kl_loss(zms[:, 0:1], zms[:, 1:2])

        loss = (rec_loss + beta * kl)

        assert loss.size() == (bs, 1), f'{loss.size()}'

        loss = loss.mean()

        loss.backward()
        opt.step()

        bar.set_postfix({
            'loss' : loss.item(),
            'kl' : kl.mean().item(),
            'ms': out[:, 1].exp().mean().item()})
        opt.zero_grad()

    print('Training done, evaluating')
    with torch.no_grad():
        N = 15000
        x = intsource(num=N, n=n)

        zms = enc(x)
        if generalized:
            z = sample(zms[:, 0:1], zms[:, 1:2])
        else:
            z = gsample(zms[:, 0:1], zms[:, 1:2].exp(), zms[:, 2:].exp())

        plt.figure()

        plt.scatter(x, z, c='b', alpha=0.5, linewidth=0, s=5)
        plt.xlabel('in')
        plt.ylabel('latent')

        plt.savefig('vae_latent.png')

        plt.figure()

        out = dec(z)
        norm = dst.Normal(out[:, :1], out[:, 1:].exp() + EPS)
        xp = norm.sample()

        plt.scatter(x, xp, c='b', alpha=0.5, linewidth=0, s=5)
        plt.xlabel('in')
        plt.ylabel('out')

        plt.savefig('vae_recon.png')

        plt.figure()


        if generalized:
            z = gsample(torch.zeros(N, 1), torch.ones(N, 1), torch.full(size=(N, 1), fill_value=prior_alpha))
        else:
            z = torch.randn(N, 1)

        plt.hist(z, bins=100)
        plt.savefig('vae_prior.png')

        out = dec(z)
        norm = dst.Normal(out[:, :1], out[:, 1:].exp() + EPS)
        xp = norm.sample()

        plt.figure()
        plt.hist(xp, bins=100)
        plt.savefig('vae_sample.png')


def test(
    bs=128,
    hidden=16,
    lr=3e-4,
    num_batches=500,
    beta=1.0,
    smult = 1e-4,
    sadd = 1.0,
):
    model = nn.Sequential(
        nn.Linear(1, 10), # no activation
        nn.Linear(10, 2)
    )

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    print(model)
    for i in (bar := trange(num_batches)):

        x = torch.randn(bs, 1)

        out = model(x)
        norm = dst.Normal(out[:, :1], (out[:, 1:] * smult + sadd).exp() + EPS)
        loss = - norm.log_prob(x) # broadcasting?
        # loss = (out[:, 0:1] - x).pow(2)

        loss.mean().backward()
        opt.step()

        bar.set_postfix({
            'loss' : loss.mean().item(),})

        opt.zero_grad()

    print(model)

def testkl():
    """
    Check if the GGD KL is the same as the Gaussian KL is we set the shape param to 2.
    :return:
    """

    N= 10_000_000
    mus = torch.tensor([0., 0., 0., 0.])
    sigs = torch.tensor([0.1, 0.5, 1., 2.])
    alphs = torch.ones_like(mus)*2

    print(kl_loss(mus, (sigs**2).log() ))
    print(nkl(mus, sigs, samples=N))
    print(gkl(mus, sigs, alphs, samples=N))

def testkl2():
    """
    Check if the GGD KL is the same as the Gaussian KL is we set the shape param to 2.
    :return:
    """


if __name__ == '__main__':
    fire.Fire()