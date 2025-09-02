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

def expand_as_right(x, y):
    """
    Expand x as y, but insert any extra dimensions at the back not the front.
    :return:
    """
    while (len(x.size()) < len(y.size())):
        x = x.unsqueeze(-1)
    return x.expand_as(y)

def kl_loss(zmean, zsig):
    b, l = zmean.size()

    kl = 0.5 * torch.sum(zsig.exp() - zsig + zmean.pow(2) - 1, dim=1)

    assert kl.size() == (b,)

    return kl

def sample(zmean, zsig):
    b, l = zmean.size()

    # sample epsilon from a standard normal distribution
    eps = torch.randn(b, l)

    # transform eps to a sample from the given distribution
    return zmean + eps * (zsig * 0.5).exp()


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape) # keep the batch dimensions, reshape the rest

class AltModel(nn.Module):

    def __init__(self, channels=(12,32,128), latent_size=128, convs=True, hs=(512,256)):
        super().__init__()

        # - channel sizes
        a, b, c = channels
        self.latent_size = latent_size

        if convs:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, a, (3, 3), padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(a, b, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(b, b, (3, 3), padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(b, c, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c, c, (3, 3), padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(4 * 4 * c, 2 * latent_size)
            )
        else:
            self.encoder = nn.Sequential(
                Reshape((3 * 32 * 32,)),
                nn.Linear(3 * 32 * 32, hs[0]), nn.ReLU(),
                nn.Linear(hs[0], hs[1]), nn.ReLU(),
                nn.Linear(hs[1], latent_size*2)
            )

        if convs:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, c * 4 * 4), nn.ReLU(),
                Reshape((c, 4, 4)),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ConvTranspose2d(c, b, (3, 3), padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ConvTranspose2d(b, a, (3, 3), padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ConvTranspose2d(a, 3, (3, 3), padding=1)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, hs[1]), nn.ReLU(),
                nn.Linear(hs[1], hs[0]), nn.ReLU(),
                nn.Linear(hs[0], 3* 32*32),
                Reshape(shape=(3, 32, 32))
            )


    def forward(self, x=None, num=None, mix=None):

        if x is None:
            z = torch.randn(size=(num, self.latent_size), device=d())
        else:
            z = self.encoder(x)

            zmean, zsig = z[:, :self.latent_size], z[:, self.latent_size:]
            kl = kl_loss(zmean, zsig)

            z = sample(zmean, zsig)

            if mix is not None:
                if type(mix) is torch.Tensor:
                    emix = expand_as_right(mix, z)

                z = emix * z + (1. - emix) * torch.randn_like(z)

        y = self.decoder(z)

        return y if x is None else (y, kl)

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
        augment_mix=None,
        augment_prob=0.5,
        augment_from=0, # Start augmenting after this many instances
        beta_temp=0.0,
        beta_weights=None,
        latent_dropouts=None,
        zdo_dynamic=False,
        zdo_range=180_000, # over how many instances to change a given z-dropout from 0 to 1
        zdo_start=180_000, # After how many instances to start the z-dropout schedule.
        loss_type='dist',
        altmodel=False,
        alt_latent=128,
        alt_convs=False,
        mid_latent=128,
        drsample=False,
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


    if not altmodel:
        model = fusion.VAE(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level,
                         mid_layers=3, mid_latent=mid_latent)

        numzs = blocks_per_level * len(unet_channels) + 1 # nr of latent connections
    else:
        model = AltModel(latent_size=alt_latent, convs=alt_convs)
        numzs = 1
        # Simpler architecture(s) for debugging.

    if zdo_dynamic:
        latent_dropouts = [1.0] * numzs
        latent_dropouts[0] = 0.0

        zdo_delta = 1/zdo_range

    if torch.cuda.is_available():
        model = model.cuda()

    if ema > -1:
        model = AveragedModel(model,
                        avg_fn=get_ema_multi_avg_fn(ema),
                        use_buffers=True)

    if dp:
        model = torch.nn.DataParallel(model)
    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    path = f'./samples_vae/'
    Path(path).mkdir(parents=True, exist_ok=True)

    runloss = 0.0
    instances_seen = 0

    curbeta = beta[0]
    beta_delta = (beta[1] - beta[0]) / (beta_sched[1] - beta_sched[0])

    zdo_last = zdo_start
    for e in range(epochs):
        # Train
        model.train()
        for i, (btch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            if torch.cuda.is_available():
                btch = btch.cuda()

            b, c, h, w = btch.size()

            if augment and instances_seen > augment_from :
                with torch.no_grad():
                    abtch = btch.clone()

                    if augment_mix is None:
                        augment_mix_ = torch.rand(size=(b,), device=d())
                    else:
                        augment_mix_ = augment_mix

                    augd, _ = model(x=btch, mix=augment_mix_, zdo=latent_dropouts)

                    sel = torch.rand(size=(b,), device=d()) < augment_prob
                    abtch[~ sel] = augd[~ sel] # augment with probability augment_prob

                    # -- We augment the data by passing it through the VAE and using a (convex) mixture of the latent
                    #    code and a random latent code. This means that there is information about the original image,
                    #    which the VAE can recover, but we also introduce artifacts that are common to the current
                    #    generation of the model
            else:
                abtch = btch

            output, kls = model(x=abtch, zdo=latent_dropouts)

            if loss_type == 'dist':
                rc_loss = ((output - btch) ** 2.0).reshape(b, -1).sum(dim=1) # Simple loss
            elif loss_type == 'bce':
                rc_loss = F.binary_cross_entropy_with_logits(output, btch, reduction='none').reshape(b, -1).sum(dim=1)
            else:
                raise

            # try continuous bernoulli?

            # kls = sum(kl.reshape(b, -1).sum(dim=-1) for kl in kls)
            if not altmodel:
                kls = torch.cat([kl.reshape(b, -1).sum(dim=-1, keepdim=True) for kl in kls], dim=1)

                if beta_weights is None:
                    weights = (kls.detach() * beta_temp).softmax(dim=-1) # -- weigh KLS proportional to relative magnitude, for
                                                                     #    adversarial setting of beta balance
                else:
                    weights = torch.tensor(beta_weights, dtype=torch.float, device=d()).unsqueeze(0).expand_as(kls)
                    weights = 10 ** weights
                    weights = weights.softmax(dim=-1)

                    if instances_seen == 0:
                        print(weights[0, :])

                assert kls.size() == weights.size()
                kls = (kls * weights).sum(dim=-1)

                loss = (rc_loss + curbeta * kls).mean()
            else:
                loss = (rc_loss + curbeta * kls).mean()

            loss.backward()

            gn = gradient_norm(model)
            if gc > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gc)

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

            if zdo_dynamic:
                if zdo_start < instances_seen < (zdo_range * (numzs-1)) + zdo_start:

                    whichz = (instances_seen - zdo_start) // zdo_range
                    for i in range(whichz+1):
                        latent_dropouts[i] = 0. # This is lazy, but WE

                    assert 0 <= whichz < numzs - 1 , f'{instances_seen} {whichz} {numzs}'

                    step = zdo_delta * (instances_seen - zdo_last)
                    latent_dropouts[whichz + 1] = max(0.0, latent_dropouts[whichz + 1] - step)

                    zdo_last = instances_seen
                    # if random.random() < 0.001:
                    print(instances_seen, latent_dropouts)

                if instances_seen > (zdo_range * (numzs-1)) + zdo_start:
                    latent_dropouts = [0.] * numzs

            if beta_sched[0] < instances_seen < beta_sched[1]:
                # curbeta = beta[0] + (beta[1] - beta[0]) * (instances_seen - beta_sched[0]) / (
                #             beta_sched[1] - beta_sched[0])

                curbeta = beta[0] + (beta[1] - beta[0]) * \
                    ((instances_seen - beta_sched[0]) / (beta_sched[1] - beta_sched[0]) ) ** beta_p

        ### Sample

        print('Generating sample, epoch', e)
        model.eval()

        with torch.no_grad():

            # 16 random samples
            ims = model(num=16, zdo=latent_dropouts) # sample 16 images
            if loss_type=='bce':
                ims = ims.sigmoid()
            griddle(ims, path + f'samples-{e}-{n:05}.png')

            # plot_at = set([2,5,10,50,100])
            # for i in range(max(plot_at)):
            #     ims, _ = model(x=ims)
            #     if loss_type == 'bce':
            #         ims = ims.sigmoid()
            #     if i in plot_at:
            #         griddle(ims, path + f'samples-{e}-{n:05}-it{i}.png')

            # Denoise/renoise algorithm
            if drsample:
                steps = 50
                plot_at = set([ 1, 2, 3, 5, 10, 20, 30, 50])

                ims = model(num=16, zdo=latent_dropouts)
                if i in plot_at:
                    griddle(ims, path + f'drsample-{e}-{n:05}-it{0}.png')

                for i in range(1, steps+1):

                    # noising
                    ims, _ = model(x=ims, mix=1 - i/steps, zdo=latent_dropouts)
                    # denoising
                    ims, _ = model(x=ims, mix=1, zdo=latent_dropouts)

                    if loss_type == 'bce':
                        ims = ims.sigmoid()

                    if i in plot_at:
                        griddle(ims, path + f'drsample-{e}-{n:05}-it{i}.png')

            # 8 examples of augmentation
            btch = btch[torch.randperm(btch.size(0))][:8]

            if augment_mix is None:
                augment_mix_ = torch.rand(size=(8,), device=d())
            else:
                augment_mix_ = augment_mix

            out, _ = model(btch, mix=augment_mix_, zdo=latent_dropouts)
            if loss_type=='bce':
                out = out.sigmoid()

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