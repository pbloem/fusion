import torch, math
from .tools import d

"""
Degradation operators
"""

def batch(ims, op, t=None, *args, **kwargs):
    """
    Apply a degradation op to a batch.

    :param ims: Batch of images
    :param op: Degradation operator (a function)
    :param t: The degradation levels. If None, each instance gets a randomly chosen level
    :return:
    """

    with torch.no_grad():

        ims = ims.clone()
        b = ims.size(0)
        t = torch.rand(size=(b, )) if t is None else t
        if type(t) == float:
            t = torch.tensor([t], device=d()).expand((b, ))

        res = []
        for i in range(b):
            op(ims[i:i+1], t[i].item(), *args, **kwargs)

        return ims

def tile(im:torch.Tensor, t:float, nh:int , nw:int, fv=0):
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
