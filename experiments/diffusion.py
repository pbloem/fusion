import fusion
from fusion.tools import here, d, fc, gradient_norm, tic, toc, prod

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

if 'wandb' in sys.modules:
    print(f"Founds weights and biases installed. Importing.")
    import wandb
else:
    wandb = None


""" 
Implementations of the basic idea behind diffusion. 
"""

def make_plots(bs = 128, k=128, steps=240):

    tic()
    dataloader, (h, w), n = data('mnist', data_dir='.', batch_size=bs,  grayscale=True, size=(32, 32))
    print(f'data loaded ({toc():.4} s)')

    Path('plots/').mkdir(parents=True, exist_ok=True)

    avg = torch.zeros(1, 3, 32, 32)
    n = 0
    for batch, _ in dataloader:
        avg += batch.sum(dim=0)
        n += batch.size(0)

    plt.imshow((avg[0]/n).clip(0, 1).permute(1, 2, 0))
    plt.gca().axis('off')
    plt.savefig(f'plots/average.png')

    # plot the data
    for batch, _ in dataloader:
        grid = make_grid(batch[16:32].clip(0, 1), nrow=4).permute(1, 2, 0)
        plt.imshow(grid)
        plt.gca().axis('off')
        plt.savefig(f'plots/data.png')

        break

    i = 36 # instance to plot

    # Gaussian noise
    gamma = bargamma = 0.98
    zt = batch.clone()
    for s in trange(20):

        plt.imshow(zt[i].permute(1, 2, 0).clip(0, 1))
        plt.gca().axis('off')
        plt.savefig(f'plots/gaussianz_{s:02}.png')

        # Sample a random binary tensor, the same size as the batch
        noise = torch.randn(size=(bs, 3, h, w))
        zt = bargamma * zt + (1-bargamma**2) * noise

        bargamma *= gamma

    zt = batch.clone()
    for s in trange(120):

        plt.imshow(zt[i].permute(1, 2, 0))
        plt.gca().axis('off')
        plt.savefig(f'plots/z_{s:02}.png')

        # Sample a random binary tensor, the same size as the batch
        noise = torch.rand(size=(bs, 1, h, w)).round().expand(bs, 3, h, w)

        # Sample `k` pixel indices to apply the noise to
        indices = torch.randint(low=0, high=32, size=(k, 2))
        # -- To keep things simple, we'll corrupt the same pixels for each image in the batch. Whether they are made
        #    black or white still differs per image.

        # Change the values of the sampled indices to those of the random tensor
        zt[:, :, indices[:, 0], indices[:, 1]] = noise[:, :, indices[:, 0], indices[:, 1]]


def naive(
        epochs=5,
        steps=60,
        name={'noname'},
        k=32,
        lr=3e-4,
        bs=16,
        sample_bs = 16,
        limit=float('inf'), # limits the number of batches per epoch,
        unet_channels=(8,16,24),
        data_name='mnist',
        data_dir=None,
        size=(32,32),
        num_workers=2,
        grayscale=False,
        res_cat=False,
        blocks_per_level=3,
        warmup = 1000,
        gc = 1.0,
        debug = False,
        disc_time = False
     ):
    """
    A Naive approach to diffussion with very little math.

    :param epochs:
    :param steps:
    :param k:
    :param lr:
    :param bs:
    :param limit:
    :param imsize:
    :return:
    """

    if wandb is not None:
        wd = wandb.init(
            name = f'naive1-{name}-{data_name}',
            project = 'diffusion',
            tags = [],
            config =locals(),
            mode = 'disabled' if debug else 'online'
        )

    tic()
    dataloader, (h, w), n = data(data_name, data_dir, batch_size=bs, nw=num_workers, grayscale=grayscale, size=size)
    print(f'data loaded ({toc():.4} s)')

    unet = fusion.UNet(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level, mid_layers=3,
                             res_cat=res_cat, max_time=steps, disc_time=disc_time)

    if torch.cuda.is_available():
        unet.cuda()

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())
    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda num_batches: min(num_batches / (warmup / bs), 1.0))

    Path('samples_naive/').mkdir(parents=True, exist_ok=True)

    for e in range(epochs):
        for i, (batch, _) in (bar := tqdm.tqdm(enumerate(dataloader), total=ceil(n/bs))):
            cbs = batch.size(0)

            if i > limit:
                break

            if torch.cuda.is_available():
                batch = batch.cuda()

            for s in range(steps):

                old_batch = batch.clone()

                if i == 0 and (s+1) % 10 == 0:
                    grid = make_grid(batch[:16].cpu(), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.savefig(f'samples_naive/noising-{s:05}.png')
                    plt.gca().axis('off')

                # Sample a random binary tensor, the same size as the batch
                noise = torch.rand(size=(cbs, 1, h, w), device=d()).round().expand(cbs, 3, h, w)

                # Sample `k` pixel indices to apply the noise to
                indices = torch.randint(low=0, high=32, size=(k, 2), device=d())
                # -- To keep things simple, we'll corrupt the same pixels for each image in the batch. Whether they are made
                #    black or white still differs per image.

                # Change the values of the sampled indices to those of the random tensor
                batch[:, :, indices[:, 0], indices[:, 1]] = noise[:, :, indices[:, 0], indices[:, 1]]

                # Train the model to denoise
                output = unet(batch, time=s if disc_time else s/steps).sigmoid()

                loss = ((output - old_batch) ** 2.0).mean()
                # loss = F.binary_cross_entropy(output, old_batch).mean()

                loss.backward()


                bar.set_postfix({'loss' : loss.item()})

                if wandb:
                    wandb.log({
                        'loss': loss,
                        'gradient_norm': gradient_norm(unet),
                        'learning_rate': sch.get_last_lr()[0],
                    })

                if gc > 0.0:
                    nn.utils.clip_grad_norm_(unet.parameters(), gc)

                opt.step()
                opt.zero_grad()

                if warmup > 0:
                    sch.step()

        with torch.no_grad():
            # Sample from the model
            batch = torch.rand(size=(sample_bs, 1, h, w), device=d()).round().expand(sample_bs, 3, h, w)
            # -- This is the distribution to which our noising process above converges
            for s in (bar := tqdm.trange(steps)):
                t = steps-s-1
                batch = unet(batch, time=t if disc_time else t/steps).sigmoid()

                if (s+1) % 10 == 0:
                    grid = make_grid(batch[:16].clip(0, 1).cpu(), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.savefig(f'samples_naive/denoising-{e}-{s:05}.png')
                    plt.gca().axis('off')

def data(name, data_dir, batch_size, nw=2, size=None, grayscale = False):

    if name == 'mnist':
        h, w = 32, 32
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [torchvision.transforms.Grayscale(num_output_channels=3),
             transforms.Resize((h, w)),
             transforms.ToTensor()])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    if name == 'mnist-128':
        h, w = 128, 128
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [torchvision.transforms.Grayscale(num_output_channels=3),
             transforms.Resize((h, w)),
             transforms.ToTensor()])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    if name == 'ffhq-thumbs':


        h, w = (128, 128) if size is None else size

        # Load MNIST and scale up to 32x32, with color channels
        transform = [] if (h, w) == (128, 128) else [transforms.Resize((h, w))]
        if grayscale:
            transform.append(torchvision.transforms.Grayscale(num_output_channels=3))
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)

        dataset = (torchvision.datasets.ImageFolder(root=data_dir, transform=transform))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    if name == 'ffhq':

        h, w = 1024, 1024
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [transforms.ToTensor()])

        dataset = (torchvision.datasets.ImageFolder(root=data_dir, transform=transform))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    fc(name)

def naive2(
        epochs = 5,
        steps = 60,
        lr = 3e-4,
        bs = 16,              # Training batch size
        sample_bs = 16,       # Sampling batch size
        limit = float('inf'), # limits the number of batches per epoch,
        algorithm2 = False,
        fix_noise = False,  # When "renoising", always use the same noise
        data_name = 'mnist',
        data_dir = None,
        unet_channels = (16, 32, 64), # Basic structure of the UNet in channels per block
        debug = False,         # Debugging mode bypasses wandb
        name = '',
        res_cat = False,
        warmup = 1000,
        gc = 1.0,
        dp = False,
        plot_renoised = False,
        plot_every = 10,       # Plots every n steps of the sampling process
        sample_every = 3,      # Sample from the model every n epochs
        num_workers = 2,       # Number fo workers for the data loader
        blocks_per_level = 3,  # Number of Res blocks per level of the UNet
        grayscale=False,       # Whether to convert the data to grayscale
        size=None,             # What to resize the data to
        shuffle=True
    ):
    """

    A second naive approach. This method trains the model to fully denoise and then samples
    by a denoising/renoising process

    :param epochs:
    :param steps:
    :param k:
    :param lr:
    :param bs:
    :param limit:
    :param imsize:
    :return:
    """

    if wandb:
        wd = wandb.init(
            name = f'naive2-{name}-{data_name}',
            project = 'diffusion',
            tags = [],
            config =locals(),
            mode = 'disabled' if debug else 'online'
        )

    scaler = torch.cuda.amp.GradScaler()

    tic()
    dataloader, (h, w), n = data(data_name, data_dir, batch_size=bs, nw=num_workers, grayscale=grayscale, size=size)
    print(f'data loaded ({toc():.4} s)')

    unet = fusion.UNet(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level, mid_layers=3,
                             res_cat=res_cat, max_time=h*w)

    if torch.cuda.is_available():
        unet = unet.cuda()

    if dp:
        unet = torch.nn.DataParallel(unet)
    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())
    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda num_batches: min(num_batches / (warmup / bs), 1.0))

    # Create an output directory to put the samples
    Path('./samples_naive2/').mkdir(parents=True, exist_ok=True)

    # Create a list containing all pixel indices for quick sampling
    indices = [(x, y) for x in range(h) for y in range(w)]
    if shuffle:
        random.shuffle(indices)

    total = len(indices)
    print('Pixel indices created.')

    for e in range(epochs):

        for i, (batch, _) in (bar := tqdm.tqdm(enumerate(dataloader), total=int(ceil(n/bs)))):
            b, c, h, w = batch.size()

            if i > limit:
                break

            if torch.cuda.is_available():
                batch = batch.cuda()

            initial_batch = batch.clone()

            t = torch.randint(low=1, high=len(indices), size=(b, ), device=d())

            if shuffle:
                random.shuffle(indices)
            batch = add_noise_var(batch, t, indices)

            # Train the model to denoise
            with torch.cuda.amp.autocast():
                output = unet(batch, time=t).sigmoid()

                loss = ((output - initial_batch) ** 2.0).mean()
                # -- We predict the _fully_ denoised batch. This will be blurry for high t, but we fix this in the sampling.

            scaler.scale(loss).backward()

            bar.set_postfix({'loss' : loss.item()})
            if wandb:
                wandb.log({
                    'loss': loss,
                    'gradient_norm': gradient_norm(unet),
                    'learning_rate': sch.get_last_lr()[0],
                })

            if gc > 0.0:
                nn.utils.clip_grad_norm_(unet.parameters(), gc)

            scaler.step(opt)
            scaler.update()

            opt.zero_grad()

            if warmup > 0:
                sch.step()

        if e % sample_every == 0:
            with ((torch.no_grad())):
                # Sample from the model

                batch = torch.rand(size=(sample_bs, 1, h, w), device=d()).round().expand(sample_bs, 3, h, w)
                # -- This is the distribution to which our noising process above converges

                noise = torch.rand(size=(sample_bs, 1, h, w), device=d()).round().expand(sample_bs, 3, h, w) \
                        if fix_noise else None;

                for s in (bar := tqdm.trange(steps)):

                    t =   int((total-1) * (steps-s)/steps)   # where we are in the denoising process from 0 to 1
                    tm1 = int((total-1) * (steps-s-1)/steps) # next noise level, t-1

                    denoised = unet(batch, time=t).sigmoid()  # denoise

                    if not fix_noise:
                        if shuffle:
                            random.shuffle(indices)

                    # -- note that the indices are _not_ shuffled, so that the noise is kept the same between steps

                    if not algorithm2:
                        batch = add_noise_var(denoised, int(tm1), indices, noise=noise) # renoise
                    else:
                        batch = batch - add_noise(denoised, int(t), indices, noise=noise) \
                                      + add_noise(denoised, int(tm1), indices, noise=noise)

                    if (s+1) % plot_every == 0:
                        grid = make_grid(denoised.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                        plt.imshow(grid)
                        plt.gca().axis('off')
                        plt.savefig(f'./samples_naive2/denoised-{e}-{s:05}.png')

                        plt.imshow(denoised[0].cpu().clip(0, 1).permute(1, 2, 0) )
                        plt.gca().axis('off')
                        plt.savefig(f'./samples_naive2/single-denoised-{e}-{s:05}.png')

                        if plot_renoised:
                            grid = make_grid(batch.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                            plt.imshow(grid)
                            plt.gca().axis('off')
                            plt.savefig(f'./samples_naive2/renoised-{e}-{s:05}.png')

                            plt.imshow(denoised[0].cpu().clip(0, 1).permute(1, 2, 0) )
                            plt.gca().axis('off')
                            plt.savefig(f'./samples_naive2/single-renoised-{e}-{s:05}.png')

def add_noise(batch, t, indices, noise=None):
    """
    Take a batch of clean images, and add noise to level t
    :param batch:
    :param indices:
    :param t:
    :return:
    """
    batch = batch.clone()

    if t == 0:
        return batch

    b, c, h, w = batch.size()

    # indices of the pixels to be corrupted.
    indt = torch.tensor(indices[:t])

    # Sample a random binary tensor, the same size as the batch
    if noise is None:
        noise = torch.rand(size=(b, 1, h, w), device=d()).round().expand(b, 3, h, w)
    # -- To keep things simple, we'll corrupt the same pixels for each image in the batch. Whether they are made
    #    black or white still differs per image.

    # Change the values of the sampled indices to those of the random tensor
    batch[:, :, indt[:, 0], indt[:, 1]] = noise[:, :, indt[:, 0], indt[:, 1]]

    return batch

def add_noise_var(batch, t, indices, noise=None):
    """
    Variable version of add noise. Allows for a different t for each instance in the batch.

    :param batch:
    :param indices:
    :param t:
    :return:
    """
    b, c, h, w = batch.size()

    if type(t) == int:
        t = torch.tensor([t])
        t = t.expand(b)

    assert t.size(0) == b, f'{t.size()} {b}'

    batch = batch.clone()

    indt = []
    for i, ti in enumerate(t):
        indt.extend((i, x, y) for x, y in indices[:ti])
        # -- These are all the indices in the batch tensor that should be corrupted. (There is room for
        #    optimization here).
    indt = torch.tensor(indt, device=d())

    if indt.numel() == 0:
        return batch

    # Sample a random binary tensor, the same size as the batch
    if noise is None:
        noise = torch.rand(size=(b, 1, h, w), device=d()).round().expand(b, 3, h, w)

    # Change the values of the sampled indices to those of the random tensor
    batch[indt[:,0], :, indt[:, 1], indt[:, 2]] = noise[indt[:,0], :, indt[:, 1], indt[:, 2]]

    return batch

def gaussian1(
        epochs=5,
        steps=120,
        lr=3e-4,
        bs=16,
        limit=float('inf'), # limits the number of batches per epoch,
        data_name='mnist',
        data_dir='./data',
        size=(32, 32),
        beta=0.9,
        gamma_sched=lambda t : 0.99,   # schedule for gamma (diffusion parameter)
        tau_sched='sigma', # schedule for tau   (model variance)
        num_workers=2,
        grayscale=False,
        dp=False,
        unet_channels=(16, 32, 64),  # Basic structure of the UNet in channels per block
        blocks_per_level=3,
        res_cat=True,
        sample_bs=16,
        plot_every=5,
        simple_loss=False,
        debug=False,
        name='gaussian1',
        algorithm=1 # 1 high variance original, 2 kl trick, 3 noise prediction

):

    """
    Simple high-variance version of the Gaussian algorithm.
    """

    h, w = size

    scaler = torch.cuda.amp.GradScaler()

    if wandb:
        wd = wandb.init(
            name = f'naive2-{name}-{data_name}',
            project = 'diffusion',
            tags = [],
            config =locals(),
            mode = 'disabled' if debug else 'online'
        )

    tic()
    dataloader, (h, w), n = data(data_name, data_dir, batch_size=bs, nw=num_workers, grayscale=grayscale, size=size)
    print(f'data loaded ({toc():.4} s)')

    gammas     = torch.tensor([gamma_sched(t) for t in range(steps)]).to(d())
    gammas_bar = torch.tensor([prod(gammas[:t]) for t in range(steps)]).to(d())
    sigmas_bar = torch.tensor([1 - gammas_bar[t]**2 for t in range(steps)]).to(d())

    if tau_sched == 'sigma':
        taus = sigmas_bar
    else:
        taus = torch.tensor([tau_sched(t) for t in range(steps)]).to(d())

    print('gammas', gammas[-10:].tolist())
    print('gammas bar', gammas_bar[-10:].tolist())
    print('sigmas bar', sigmas_bar[-10:].tolist())

    unet = fusion.UNet(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level, mid_layers=3,
                             res_cat=res_cat, max_time=h*w)

    if torch.cuda.is_available():
        unet = unet.cuda()

    if dp:
        unet = torch.nn.DataParallel(unet)
    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())

    pth = f'./samples_gaussian_{algorithm}/'
    Path(pth).mkdir(parents=True, exist_ok=True)

    for e in range(epochs):
        # Train
        unet.train()
        for i, (batch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            if torch.cuda.is_available():
                batch = batch.cuda()

            b, c, h, w = batch.size()

            t = torch.randint(low=1, high=steps, size=(b,), device=d())

            m = torch.ones_like(output) if simple_loss else (1.0 / (2.0 * taus[t]))[:, None, None, None]
            if algorithm == 1:
                noise_t = torch.randn(size=(b, c, h, w), device=d())
                noise_tm1 = torch.randn(size=(b, c, h, w), device=d())

                ztm1 = gammas_bar[t-1, None, None, None] * batch + sigmas_bar[t-1, None, None, None].sqrt() * noise_tm1
                g = gammas[t, None, None, None]
                zt   = g * ztm1   + (1 - g ** 2).sqrt() * noise_t

                assert zt.size() == (b, c, h, w)

                # The model predicts the previous z
                output = unet(zt, time=t).sigmoid()

                loss = (m * (output - ztm1) ** 2.0).mean()

            elif algorithm == 2:

                noise = torch.randn(size=(b, c, h, w), device=d())
                gt, gbt = gammas[t].item(), gammas_bar[t].item()

                mutilde = (1/gt) * (batch - ((1-gt**2) / math.sqrt(1-gbt)) * noise)

                # The model predicts the mean for the previous zt (mutilde)
                output = unet(zt, time=t).sigmoid()
                loss = (m * (mutilde - ztm1) ** 2.0).mean()

            else:
                fc(algorithm, 'algorithm')

            loss.backward()
            opt.step()
            opt.zero_grad()

            bar.set_postfix({'loss' : loss.item()})

        # Sample
        unet.eval()
        with torch.no_grad():

            batch = torch.randn(size=(sample_bs, c, h, w), device=d())

            for t in trange(steps-1, 0, -1):

                if algorithm in (1, 2):

                    pred = unet(batch, t).sigmoid()
                    batch = dst.Normal(pred, taus[t]).sample()

                    if (t + 1) % plot_every == 0:
                        grid = make_grid(batch.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                        plt.imshow(grid)
                        plt.gca().axis('off')
                        plt.savefig(f'{pth}denoised-{e}-{t:05}.png')

                        grid = make_grid(pred.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                        plt.imshow(grid)
                        plt.gca().axis('off')
                        plt.savefig(f'{pth}mean-{e}-{t:05}.png')

                else:
                    fc(algorithm)

def gaussian(
        epochs=5,
        steps=120,
        lr=3e-4,
        bs=16,
        limit=float('inf'), # limits the number of batches per epoch,
        data_name='mnist',
        data_dir='./data',
        size=(32, 32),
        beta=0.9,
        gamma_sched=lambda t : 0.9,   # schedule for gamma (diffusion parameter)
        tau_sched=lambda t, s: 0.1 * (t/s), # schedule for tau   (model variance)
        num_workers=2,
        grayscale=False,
        dp=False,
        unet_channels=(16, 32, 64),  # Basic structure of the UNet in channels per block
        blocks_per_level=3,
        res_cat=True,
        sample_bs=16,
        plot_every=5,
):

    """
    Gaussian diffusion with constant parameters. That every noising step has

    $$z_{t+1} ~ N(\beta z_{t}, \sigma I)$$

    with fixed constants $\beta$ and $\sigma$. To ensure that the process converges to a standard normal distribution
    we set $0 < \beta <1$ and $\sigma = 1 - \beta^2$.
    """

    h, w = size

    scaler = torch.cuda.amp.GradScaler()

    tic()
    dataloader, (h, w), n = data(data_name, data_dir, batch_size=bs, nw=num_workers, grayscale=grayscale, size=size)
    print(f'data loaded ({toc():.4} s)')

    gammas     = torch.tensor([gamma_sched(t) for t in range(steps)]).to(d())
    gammas_bar = torch.tensor([prod(gammas[:t]) for t in range(steps)]).to(d())
    sigmas_bar = torch.tensor([1 - gammas_bar[t]**2 for t in range(steps)]).to(d())

    unet = fusion.UNet(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level, mid_layers=3,
                             res_cat=res_cat, max_time=h*w)

    if torch.cuda.is_available():
        unet = unet.cuda()

    if dp:
        unet = torch.nn.DataParallel(unet)
    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())

    Path('./samples_gaussian/').mkdir(parents=True, exist_ok=True)

    for e in range(epochs):
        # Train
        unet.train()
        for i, (batch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            if torch.cuda.is_available():
                batch = batch.cuda()

            b, c, h, w = batch.size()

            t = torch.randint(low=1, high=steps, size=(b,), device=d())
            noise = torch.randn(size=(b, c, h, w), device=d())

            zt = gammas_bar[t, None, None, None] * batch + sigmas_bar[t, None, None, None].sqrt() * noise
            assert zt.size() == (b, c, h, w)

            # The model predicts the noise vector
            output = unet(zt, time=t).sigmoid()
            loss = ((output - noise) ** 2.0).mean() # Simple loss

            loss.backward()
            opt.step()
            opt.zero_grad()

            bar.set_postfix({'loss' : loss.item()})

        # Sample
        unet.eval()
        with torch.no_grad():

            batch = torch.randn(size=(sample_bs, c, h, w), device=d())

            for t in trange(steps-1, 0, -1):
                pred = unet(batch, t).sigmoid()
                gt, gbt = gammas[t].item(), gammas_bar[t].item()

                mutilde = (1/gt) * (batch - ((1-gt**2) / math.sqrt(1-gbt)) * pred )
                # std = tau_sched(t)
                std = sigmas_bar[t]

                assert std > 10e-16
                batch = dst.Normal(mutilde, std).sample()

                if (t + 1) % plot_every == 0:
                    grid = make_grid(batch.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.gca().axis('off')
                    plt.savefig(f'./samples_gaussian/denoised-{e}-{t:05}.png')

                    grid = make_grid(mutilde.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.gca().axis('off')
                    plt.savefig(f'./samples_gaussian/mean-{e}-{t:05}.png')


if __name__ == '__main__':
    fire.Fire()