import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import os, time, math


tics = []

def tic():
    tics.append(time.time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()

def log2(x):
    return math.log(x) / math.log(2.0)

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def packdir(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the outer 'clearbox' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def here(file, subpath=None):
    """

    The path in which the given file resides, or a path relative to it if subpath is provided.

    Call with here(__file__) to get a path relative to the current executing code.

    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(file)))

    return os.path.abspath(os.path.join(os.path.dirname(file), subpath))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def logsumexp(array, axis, keepdims=False):
    """
    Numerically stable log-sum over
    :param array:
    :param axis:
    :return:
    """

    ma = array.max(axis=axis, keepdims=True)
    array = np.log(np.exp(array - ma).sum(axis=axis, keepdims=keepdims))
    array = array + (ma if keepdims else ma.squeeze(axis))

    return array

def kl_categorical(p, q, eps=1e-10):
    """
    Computes KL divergence between two categorical distributions.

    :param p: (..., n)
    :param q: (..., n)
    :return:
    """

    p = p + eps
    q = q + eps

    # entropy of p
    entp = - p * np.log2(p)
    entp = entp.sum(axis=-1)

    # cross-entropy of p to q
    xent = - p * np.log2(q)
    xent = xent.sum(axis=-1)

    return xent - entp

def flatten(tensors):
    """
    Flattens an iterable over tensors into a single vector
    :param tensors:
    :return:
    """
    return torch.cat([p.reshape(-1) for p in tensors], dim=0)

def prod(tuple):
    res = 1
    for v in tuple:
        res *= v
    return res

def set_parameters(parameters, model):
    """
    Take the given vector of parameters and set them as the parameters of the model, slicing and reshaping as necessary.

    :param params:
    :param model:
    :return:
    """
    if parameters is not None:

        cursor = 0
        for p in model.parameters():
            size = prod(p.size())
            slice = parameters[cursor:cursor + size]
            slice = slice.reshape(p.size())
            p.data = slice.contiguous()

            cursor = cursor + size

def coords(h, w):
    """
    Generates a pixel grid of coordinate representations for the given width and height (in the style of the coordconv).
    The values are scales to the range [0, 1]

    This is a cheap alternative to the more involved, but more powerful, coordinate embedding or encoding.

    :param h:
    :param w:
    :return:
    """
    xs = torch.arange(h, device=d())[None, None, :, None] / h
    ys = torch.arange(w, device=d())[None, None, None, :] / w
    xs, ys = xs.expand(1, 1, h, w), ys.expand(1, 1, h, w)
    res = torch.cat((xs, ys), dim=1)

    assert res.size() == (1, 2, h, w)

    return res


class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Debug(nn.Module):
    def __init__(self, lambd):
        super(Debug, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        self.lambd(x)
        return x

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class NoActivation(nn.Module):
    def forward(self, input):
        return input

def fc(var, name=None):
    if name is not None:
        raise Exception(f'Unknown value {var} for variable with name {name}.')
    raise Exception(f'Unknown value {var}.')

