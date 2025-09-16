import torch, fire

import fusion
from fusion import tile, batch, fourier
from fusion.tools import here, d, fc, gradient_norm, tic, toc, prod, data

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from torch import fft
import torch.nn.functional as F

def deg(data_name='mnist', data_dir='./data', bs=16, t=0.6, fv=0, aa=32):
    """

    :param data_name:
    :param data_dir:
    :param bs:
    :param t:
    :param fv:
    :param aa: Amount of anti-aliasing (of the mask)
    :return:
    """

    tic()
    dataloader, (h, w), n = data(data_name, data_dir, batch_size=bs, nw=1, grayscale=False)
    print(f'data loaded ({toc():.4} s)')

    ims, _ = next(iter(dataloader))

    griddle(ims, 'images_in.png')

    ims = fft.fft2(ims)
    griddle(ims.abs(), 'images_fft.png')

    # mh, mw = int(h // 2 * t), int(w // 2 * t)  # -- distance from center to mask
    # ims[:, :, h // 2 - mh:h // 2 + mh, w // 2 - mw:w // 2 + mw] = fv
    crds = coords(h * aa, w *aa)
    dists = (crds[:, 0, :, :] ** 2 + crds[:, 1, :, :] ** 2).sqrt()
    thresh = dists.max() * t
    mask = 1.0 - (dists < thresh).to(torch.float)[:,None,:,:]
    mask = F.interpolate(mask, size=(h, w), mode='area')

    plotim(mask[0], plt.gca())
    plt.savefig('mask.png')

    ims = ims * mask.expand_as(ims)

    griddle(ims.abs(), 'images_fft_masked.png')

    ims = fft.ifft2(ims)
    griddle(ims.real, 'images_out.png')

def plotim(im, ax):
    ax.imshow(im.permute(1, 2, 0).cpu().clip(0, 1))
    ax.axis('off')

def griddle(btch, file):
    grid = make_grid(btch.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
    plt.imshow(grid)
    plt.gca().axis('off')
    plt.savefig(file)

def coords(h, w):
    """
    Generates a pixel grid of coordinate representations for the given width and height (in the style of the coordconv).
    The values are scales to the range [0, 1]

    This is a cheap alternative to the more involved, but more powerful, coordinate embedding or encoding.

    :param h:
    :param w:
    :return:
    """
    xs = torch.arange(-h//2, h//2, device=d())[None, None, :, None] + 0.5
    ys = torch.arange(-w//2, w//2, device=d())[None, None, None, :] + 0.5
    xs, ys = xs.expand(1, 1, h, w), ys.expand(1, 1, h, w)
    res = torch.cat((xs, ys), dim=1)

    assert res.size() == (1, 2, h, w)

    return res


if __name__ == '__main__':
    fire.Fire(deg)