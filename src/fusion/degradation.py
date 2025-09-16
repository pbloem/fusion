import torch, math
from .tools import d

from torch import fft
import torch.nn.functional as F

"""
Degradation operators
"""

def batch(ims, op, t=None, inplace=None, *args, **kwargs):
    """
    Apply a degradation op to a batch.

    :param ims: Batch of images
    :param op: Degradation operator (a function)
    :param t: The degradation levels. If None, each instance gets a randomly chosen level
    :return:
    """

    if inplace is None:
        assert op in [tile, fourier]

        if op in [tile]:
            inplace = True
        elif op in [fourier]:
            inplace = False

    with torch.no_grad():

        ims = ims.clone()
        b = ims.size(0)
        t = torch.rand(size=(b, )) if t is None else t
        if type(t) == float:
            t = torch.tensor([t], device=d()).expand((b, ))

        res = []
        for i in range(b):

            r = op(ims[i:i+1], t[i].item(), *args, **kwargs)
            if not inplace:
                ims[i:i+1] = r

        return ims

def tile(im:torch.Tensor, t:float, nh:int , nw:int, fv=0, *args, **kwargs):
    """
    Fills the image from left to right and top to bottom with black squares.

    :param im: A batch of images. They will al be degraded to the same level
    :param t: The level of degradation (a value between 0 and 1)
    :param nh: The number of tiles to slice horizontally
    :param nw: The number of tiles to slice vertically
    :param fv: fill value, what to set the blanked out tiles to.
    :return:
    """

    b, c, h, w = im.size()

    assert h % nh == 0 and w % nw == 0

    ph, pw = h//nh, w//nw # resolution of each tile

    # convert `t` to number of tiles
    total = nh * nw
    t = int(round(t * total))

    rows, rem = t // nw, t % nw # nr of rows (of tiles) to fully black out, remaining tiles

    im[:, :, :rows*ph, :] = fv
    im[:, :, rows*ph:(rows+1)*ph, :rem*ph] = fv

def fourier(im:torch.Tensor, t:float, aa=32, gamma=1/3, *args, **kwargs):
    """
    Degrades an image by applying a 2D FFT, removing the `t` lowest proportion of frequencies
    and inverting the fft again. 
    
    :param im: 
    :param t:
    :param gamma: t <- t ** gamma.
    :param scale: remap t's
    :return: 
    """
    b, c, h, w = im.size()

    im_fftd = fft.fft2(im)

    crds = coords(h * aa, w *aa)
    dists = (crds[:, 0, :, :] ** 2 + crds[:, 1, :, :] ** 2).sqrt()
    thresh = dists.max() * (t ** gamma)
    mask = 1.0 - (dists < thresh).to(torch.float)[:,None,:,:]
    mask = F.interpolate(mask, size=(h, w), mode='area')

    im_masked = im_fftd * mask.expand_as(im_fftd)

    return fft.ifft2(im_masked).real

def coords(h, w):
    """
    Generates a pixel grid of coordinate representations for the given width and height (in the style of the coordconv).
    The center pixel is at 0.0 (for odd resolutins)

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


