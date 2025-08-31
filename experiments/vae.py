import fusion
from fusion import tile, batch
from fusion.tools import here, d, fc, gradient_norm, tic, toc, prod, data

import fire, math, os, contextlib
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dst

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

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

import optuna

GAMMA = 0.9

def train(
        epochs=5,
        lr=3e-4,
        bs=48,
        limit=float('inf'), # limits the number of batches per epoch,
        data_name='mnist',
        data_dir='./data',
        size=(32,32),
        num_workers=2,
        grayscale=False,
        dp=False,
        unet_channels=(16,32,64),  # Basic structure of the UNet in channels per block
        blocks_per_level=3,
        sample_bs=16,
        plot_every=1,
        beta=(0.0,1.0),
        beta_sched=(0, 100),
        beta_p=1.0,  # exponent of the beta schedule (> 1 smooths out the initial values)
        name='vae',
        debug=False,
        gc=1.0,
        ema=-1,
        augment=False,
        augment_mix=0.5,
        augment_prob=1.0,
        augment_from=120_000, # Start augmenting after this many instances
):

    """
    StyleVAE with augmentation.
    """

    # Print pwd, and add to locals()
    pwd = os.getcwd()
    print('pwd', pwd)

    if wandb is not None:
        wd = wandb.init(
            name = name,
            project = 'StyleVAE',
            tags = [],
            config = locals(),
            mode = 'disabled' if debug else 'online'
        )

    h, w = size

    scaler = torch.cuda.amp.GradScaler()

    tic()
    dataloader, (h, w), n = data(data_name, data_dir, batch_size=bs, nw=num_workers, grayscale=grayscale, size=size)
    print(f'data loaded ({toc():.4} s)')

    unet = fusion.VAE(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level,
                         mid_layers=3)

    if torch.cuda.is_available():
        unet = unet.cuda()

    if ema > -1:
        unet = AveragedModel(unet,
                        avg_fn=get_ema_multi_avg_fn(ema),
                        use_buffers=True)

    if dp:
        unet = torch.nn.DataParallel(unet)
    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())

    path = f'./samples_vae/'
    Path(path).mkdir(parents=True, exist_ok=True)

    runloss = 0.0
    instances_seen = 0

    curbeta = beta[0]
    beta_delta = (beta[1] - beta[0]) / (beta_sched[1] - beta_sched[0])

    for e in range(epochs):
        # Train
        unet.train()
        for i, (btch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            if torch.cuda.is_available():
                btch = btch.cuda()

            b, c, h, w = btch.size()

            if augment and instances_seen > augment_from :
                with torch.no_grad():
                    abtch = btch.clone()
                    augd, _ = unet(x=btch, mix=augment_mix)

                    sel = torch.rand(size=(b,), device=d()) < augment_prob
                    abtch[~ sel] = augd[~ sel] # augment with probability augment_prob

                    # -- We augment the data by passing it through the VAE and using a (convex) mixture of the latent
                    #    code and a random latent code. This means that there is information about the original image,
                    #    which the VAE can recover, but we also introduce artifacts that are common to the current
                    #    generation of the model

            output, kls = unet(x=abtch)

            rc_loss = ((output - btch) ** 2.0).reshape(b, -1).sum(dim=1) # Simple loss
            # try continuous bernoulli?

            kls = sum(kl.reshape(b, -1).sum(dim=-1) for kl in kls)

            loss = (rc_loss + curbeta * kls).mean()

            loss.backward()

            gn = gradient_norm(unet)
            if gc > 0.0:
                nn.utils.clip_grad_norm_(unet.parameters(), gc)

            opt.step()

            if wandb:
                wandb.log({
                    'rec_loss': rc_loss.mean().item(),
                    'loss': loss.item(),
                    'kl_loss': sum(kls).mean().item(),
                    'gradient_norm': gn,
                    'beta': curbeta,
                })

            runloss += runloss * (1-GAMMA) + loss * GAMMA

            bar.set_postfix({'running loss' : loss.item()})
            opt.zero_grad()

            instances_seen += b

            if beta_sched[0] < instances_seen < beta_sched[1]:
                # curbeta = beta[0] + (beta[1] - beta[0]) * (instances_seen - beta_sched[0]) / (
                #             beta_sched[1] - beta_sched[0])

                curbeta = beta[0] + (beta[1] - beta[0]) * \
                    ((instances_seen - beta_sched[0]) / (beta_sched[1] - beta_sched[0]) ) ** beta_p

        ### Sample

        print('Generating sample, epoch', e)
        unet.eval()

        with torch.no_grad():

            # 16 random samples
            ims = unet(num=16) # sample 16 images
            griddle(ims, path + f'samples-{e}-{n:05}.png')

            # 8 examples of augmentation
            btch = btch[torch.randperm(btch.size(0))][:8]
            out, _ = unet(btch, mix=augment_mix)
            ims = torch.cat([btch, out], dim=0)
            griddle(ims, path + f'augment-{e}-{n:05}.png', nrow=8)

    return {'last loss' : loss, 'ema loss': runloss}

def plotim(im, ax):
    ax.imshow(im.permute(1, 2, 0).cpu().clip(0, 1))
    ax.axis('off')

def griddle(btch, file, nrow=4):
    grid = make_grid(btch.cpu().clip(0, 1), nrow=nrow).permute(1, 2, 0)
    plt.imshow(grid)
    plt.gca().axis('off')
    plt.savefig(file)

def tune_vae(trial):

    res = train(
        epochs=5,
        bs=256,
        lr=trial.suggest_float('lr', 1e-5, 2e-3, log=True),
        beta=trial.suggest_float('beta', 1e-8, 1e8, log=True),
        sched=trial.suggest_categorical('sched', ['uniform', 'discrete']),
        p=trial.suggest_float('p', 1e-2, 1e2, log=True),
        dres=4,
        id=trial.number
    )

    return res['ema loss']

def tune(trials=100, name='vcd-tune'):

    study = optuna.create_study(
        storage=f'sqlite:///db.sqlite3',  # Specify the storage URL here.
        study_name=name,
        load_if_exists=True,
        direction="minimize",
    )

    study.optimize(lambda t : tune_vcd(trial=t), n_trials=trials)

    print(f'Finished. Result:')
    print('\t', study.best_params)

if __name__ == '__main__':
    fire.Fire()