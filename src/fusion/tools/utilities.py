import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as dst
from torch import lgamma

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

def kl_loss(zmean, zsig):
    """
    Compute the KL loss between the given diagonal Gaussian and the standard Gaussian

    :param zmean:
    :param zsig: The log of the variance
    :return:
    """
    size = zmean.size()

    kl = 0.5 * (zsig.exp() - zsig + zmean.pow(2) - 1)
    # kl = kl.reshape(b, -1).sum(dim=1)

    assert kl.size() == size

    return kl

def sample(zmean, zsig):
    """
    Sample from the given diagonal Gaussian

    :param zmean:
    :param zsig:
    :return:
    """
    assert zmean.size() == zsig.size()

    # sample epsilon from a standard normal distribution
    eps = torch.randn_like(zmean)

    # transform eps to a sample from the given distribution
    return zmean + eps * (zsig * 0.5).exp()

def gamma(x):
    return torch.lgamma(x).exp()

def A(alpha, sigma):
    """
    The A function from Nardon & Pianca
    :param alpha:
    :param sigma:
    :return:
    """
    return  (1 / sigma) * (gamma(3 / alpha) / gamma(1 / alpha)).sqrt()

def t(x):
    return torch.tensor([x], device=d(), dtype=torch.float)

def ex(x, s):
    if x.numel() == 1:
        return x.expand(*s)
    return x

def nkl(mu1, sig1, mu2=t(0.0), sig2=t(1.0), samples=1):
    """
    Estimate the KL divergence between two normal distributions

    :return: The kl divergence: a tensor in the same shape as the inputs.
    """
    assert mu1.size() == sig1.size()

    mu2, sig2 = ex(mu2, mu1.size()), ex(sig2, sig1.size())

    # take n samples from the first distribution
    # compute the log of the density ratio
    norm1 = dst.Normal(mu1, sig1)

    x = norm1.sample( (samples,) ).transpose(0, 1)
    # -- NB: This transpose only works with one batch dim

    logp1 = nlogp(x, mu1[:, None], sig1[:, None])
    logp2 = nlogp(x, mu2[:, None], sig2[:, None])

    return - (logp2 - logp1).mean(dim=-1)


def gkl(mu1, sig1, alpha1, mu2=t(0.), sig2=t(1.), alpha2=t(2.), samples=1):
    """
    Estimate the kl divergence between a given Generalized normal and a prior standard normal with
    mean 0, std 1 and the given alpha value.

    Tensor shapes should match.

    NB: We use the parametrization from Nardon & Pianca, which differs from the version currently
        shown on wikipedia

    :return: The kl divergence: a tensor in the same shape as the inputs.
    """
    assert mu1.size() == sig1.size() == alpha1.size()

    mu2, sig2, alpha2 = ex(mu2, mu1.size()), ex(sig2, sig1.size()), ex(alpha2, alpha1.size())

    # take n samples from the first distribution
    # compute the log of the density ratio
    x = gsample(mu1, sig1, alpha1, n=samples, squeeze=False)

    logp1 = glogp(x, mu1[:, None], sig1[:, None], alpha1[:, None])
    logp2 = glogp(x, mu2[:, None], sig2[:, None], alpha2[:, None])

    return - (logp2 - logp1).mean(dim=-1)

def nlogp(x, mu, sig):

    t1 = - (math.sqrt(2 * math.pi) * sig).log()

    t2 = - ((x - mu) ** 2) / (2 * sig ** 2)

    return t1 + t2

def glogp(x, mu, sig, alpha):
    """
    The log probability density of `x` under the given generalized normal distribution

    :param x:
    :param mu:
    :param sig:
    :param alpha:
    :return:
    """
    afun = A(alpha, sig)

    t1 = torch.log(alpha) - torch.log(torch.tensor([2.0]))

    t2 = afun.log() - lgamma(1/alpha)

    t3 = - (afun * (x - mu).abs()) ** alpha

    return t1  + t2 + t3

def gsample(mu, sig, alpha, n=1, squeeze=True):
    """
    Take a reparametrized sample from a given Generalized normal distribution.

    Uses the algorithm from Nardon & Pianca, 2008. This involved one random binary choice, but the
    resulting pass-through estimate of the gradient should be an unbiased estimate of the true variance
    (TODO: prove).

    :param mu:
    :param sig:
    :param alpha:
    :param n: number of samples per instance (a new dimension is created)
    :param squeeze: Whether to remove the sample dimension if n=1
    :return:
    """
    assert mu.size() == sig.size() == alpha.size()
    s = mu.size()

    mu, sig, alpha = mu[..., None], sig[..., None], alpha[..., None]
    mu, sig, alpha = mu.expand(*s, n), sig.expand(*s, n), alpha.expand(*s, n)

    a = 1 / alpha
    b = A(alpha, 1).pow(alpha)

    gd = dst.Gamma(a, b)
    z = gd.rsample()
    y = z.pow(1 / alpha)
    x = y * (torch.rand(size = mu.size()).round() * 2 - 1)
    # -- x is now a mean centered, unit-std sample with shape alpha.

    res = x * sig + mu # map to the required mean and std.

    if squeeze and n == 1:
        return res.squeeze(-1)
    return res
