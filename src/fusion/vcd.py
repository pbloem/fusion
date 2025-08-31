import torch
from torch import nn
import torch.nn.functional as F

from .tools import coords, d, Lambda, sample, kl_loss


"""
Variational cold diffusion (experimental model)
"""

class ResBlock(nn.Module):
    """
    Simplified residual block. Applies a convolution, and a residual connection around it. Fixed number of channels.

    """

    def __init__(self, in_channels, out_channels, dropout=0.1, time_emb=512, coord_hidden=256, vout=False):
        """
        :param channels:
        :param dropout:
        """

        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.resconv = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        # -- pointwise conv on the residual

        self.convolution = nn.Sequential(
            nn.GroupNorm(1, in_channels), # Equivalent to LayerNorm, but over the channel dimension of an image
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # Projects the time scalars up/down to the number of input channels
        self.embed_time = nn.Linear(time_emb, in_channels)

        # Projects the pixel coordinates up to the number of channels
        self.embed_coords = nn.Sequential(
            nn.Conv2d(2, coord_hidden, 1, padding=0), nn.ReLU(),
            nn.Conv2d(coord_hidden, coord_hidden, 1, padding=0), nn.ReLU(),
            nn.Conv2d(coord_hidden, in_channels, 1, padding=0)
        )

        # variational output
        self.vout = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1, padding=0) if vout else None
        # [-- We give the vout just one output channel at the current resolution (or rather, a mean and var, so
        #    two channels, but they represent a dist on one).]

    def forward(self, x, time):
        """
        :param x: The input batch. The residual connection has already been added/concatenated
        :param time: 1-D batch of values in [0, 1] representing the time step along the noising trajectory.
        :return:
        """

        b, c, h, w = x.size()
        assert c == self.in_channels, f'{c} {self.in_channels}'

        # assert len(time.size()) == 1
        # if time.size(0) == 1:
        #     time = time.expand(b)
        # assert time.size(0) == b

        # Project time up to the number of channels ...
        time = self.embed_time(time)
        # ... and expand to the size of the image
        time = time[:, :, None, None].expand(b, c, h, w)

        # Generate a grid of coordinate representations ...
        crds = coords(h, w).expand(b, 2, h, w)
        # ... and project up to the number of channels
        crds = self.embed_coords(crds)

        # Apply the convolution and the residual connection
        res = self.resconv(x)
        res = self.convolution(x + time + crds) + res

        if self.vout is not None:
            return res, self.vout(res)
        return res


class ResBlockNT(nn.Module):
    """
    Residual block without time embedding
    """

    def __init__(self, in_channels, out_channels, dropout=0.1, coord_hidden=256, vout=False):
        """
        :param channels:
        :param dropout:
        """

        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.resconv = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        # -- pointwise conv on the residual

        self.convolution = nn.Sequential(
            nn.GroupNorm(1, in_channels), # Equivalent to LayerNorm, but over the channel dimension of an image
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # Projects the pixel coordinates up to the number of channels
        self.embed_coords = nn.Sequential(
            nn.Conv2d(2, coord_hidden, 1, padding=0), nn.ReLU(),
            nn.Conv2d(coord_hidden, coord_hidden, 1, padding=0), nn.ReLU(),
            nn.Conv2d(coord_hidden, in_channels, 1, padding=0)
        )

        # variational output
        self.vout = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1, padding=0) if vout else None
        # [-- We give the vout just one output channel at the current resolution (or rather, a mean and var, so
        #    two channels, but they represent a dist on one).]

    def forward(self, x):
        """
        :param x: The input batch. The residual connection has already been added/concatenated
        :param time: 1-D batch of values in [0, 1] representing the time step along the noising trajectory.
        :return:
        """

        b, c, h, w = x.size()
        assert c == self.in_channels, f'{c} {self.in_channels}'

        # Generate a grid of coordinate representations ...
        crds = coords(h, w).expand(b, 2, h, w)
        # ... and project up to the number of channels
        crds = self.embed_coords(crds)

        # Apply the convolution and the residual connection
        res = self.resconv(x)
        res = self.convolution(x + crds) + res

        if self.vout is not None:
            return res, self.vout(res)
        return res

class VCUNet(nn.Module):

    """
    UNet for variational cold diffusion. Adds a separate encoder branch which has access to the
    target image.

    """

    def __init__(self,
            res,
            channels = (8, 16, 32), # Number of channels at each level of the UNet
            num_blocks = 3,         # Number of res blocks per level
            mid_layers = 3,         # Number of linear layers in the middle block
            time_emb=512,           # Time embedding dimension
        ):
        super().__init__()

        self.channels = channels
        self.num_blocks = num_blocks

        # Initial convolution up to the first res block
        self.initial = nn.Conv2d(3, channels[0], kernel_size=1, padding=0)

        self.encoder   = nn.ModuleList()
        self.vcencoder = nn.ModuleList() # Copy of the encoder branch

        for i, ch in enumerate(channels):
            # Add a sequence of ResBlocks

            self  .encoder.extend(ResBlock(in_channels=ch, out_channels=ch, time_emb=time_emb)
                                  for _ in range(num_blocks))
            self.vcencoder.extend(ResBlock(in_channels=ch*2, out_channels=ch, time_emb=time_emb, vout=True)
                                  for _ in range(num_blocks))
            # -- The vc blocks get double the input channels, because they receive the output of the
            #    corresponding encoder block catted to their input.

            # Downsample
            self  .encoder.append(nn.AvgPool2d(kernel_size=2))
            self.vcencoder.append(nn.AvgPool2d(kernel_size=2))

            if i < len(channels) - 1:
                # Project up to next number of channels
                self  .encoder.append(nn.Conv2d(ch, channels[i+1], kernel_size=1, padding=0))
                self.vcencoder.append(nn.Conv2d(ch, channels[i+1], kernel_size=1, padding=0))

        m = 2 ** len(channels)
        self.mres = res[0] // m, res[1] // m
        h = channels[-1] * self.mres[0] * self.mres[1]

        print(' -- unet: midblock hidden dim:', h)

        midblock = []
        for i in range(mid_layers):
            midblock.append(nn.Linear(h, h))
            if i < mid_layers - 1:
                midblock.append(nn.ReLU())
        self.midblock = nn.Sequential(*midblock)

        rchannels = channels[::-1]
        self.decoder = nn.ModuleList()
        for i, ch in enumerate(rchannels):

            # Upsample
            self.decoder.append(
                Lambda(lambda x : F.interpolate(x, scale_factor=2, mode='nearest'))
            )

            # Add a sequence of ResBlocks
            self.decoder.extend(ResBlock(in_channels=ch * 3, out_channels=ch, time_emb=time_emb)
                                for _ in range(num_blocks))
            # -- In channels: `ch` from the previous block, `ch` from the res connection and `ch` from the
            #    vc connection

            if i < len(channels) - 1:
                # Project down to next number of channels
                self.decoder.append(nn.Conv2d(ch, rchannels[i+1], kernel_size=1, padding=0))

        # Final convolution down to the required number of output channels
        self.final = nn.Conv2d(channels[0] + 3, 3, kernel_size=1, padding=0)

        # Continuous time (0, 1), use a nonlinear function to project up
        self.timeembs = nn.Sequential(
            nn.Linear(2, time_emb * 2), nn.ReLU(),
            nn.Linear(time_emb * 2, time_emb * 2), nn.ReLU(),
            nn.Linear(time_emb * 2, time_emb),
        )

    def forward(self, x1, t0, t1, x0=None, mix_latent=1.0):
        """

        :param x1: The input to the network. The image corrupted to t1.
        :param x0: the target of the network. The image corrupted to t0 (that is, to a lesser degree
           than x1). If None, the network will run in sampling mode, skipping the variational encoder
           branch. If given, the image will be encoded into a Gaussian distribution on the latent space.
        :param t0: The timestamp corresponding to x0
        :param t1: The timestamp corresponding to x1
        :return: If x0 is None, a (diff) prediction for x0. If x0 is given, a pair consisting of a (diff) prediction for
           x0 and the kl losses for the latent distributions.
        """

        x = x1
        b, c, h, w = x.size()

        assert t0.size() == (b, ) and t1.size() == (b, ), f'{b=}, {t0.size()=}, {t1.size()=}'

        time = self.timeembs(torch.cat([t0[:, None], t1[:, None]], dim=1))

        hs = [x] # Collect values for skip connections, start with the input image
        if x0 is not None: zs = []

        x = self.initial(x) # Project up to the first nr. of channels
        if x0 is not None: xvc = self.initial(x0)

        # Encoder branches
        for mod, vcmod in zip(self.encoder, self.vcencoder):
            if type(mod) == ResBlock:

                x = mod(x, time=time)
                hs.append(x) # skip connections

                if x0 is not None:
                    xvc, z = vcmod(torch.cat([xvc, x], dim=1), time=time)
                    # -- the vc layer takes the result of the corresponding regular layer into account
                    #    this allows it to easily compute the differences between z_{t-1} and z_t
                    # -- The second output is the mean/var on the latent space.

                    zs.append(z) # encoded target output

            else:
                x = mod(x)
                if x0 is not None: xvc = vcmod(xvc)

        # Mid blocks
        x = x.reshape(b, -1) # flatten

        x = self.midblock(x) + x
        x = x.reshape(b, -1, *self.mres)

        # Decoder branch
        if x0 is not None: kl_losses = []

        for mod in self.decoder:

            if type(mod) == ResBlock:

                h = hs.pop() # The value from the relevant skip connection
                c = h.size(1)
                # print('h', h.size())

                b, _, height, width = h.size()
                if x0 is None: # sample
                    z = torch.randn(size=(b, c, height, width), device=d()) # sample from the standard Gaussian
                else:
                    z = zs.pop() # The latent from the VC encoder branch

                    assert z.size(1) == 2 * c
                    # zc = z.size(1)

                    kl_losses.append(kl_loss(z[:, :c, :, :], z[:, c:, :, :]))
                    z = sample(z[:, :c, :, :], z[:, c:, :, :])

                    if mix_latent < 1.0:
                        zp = torch.randn(size=(b, c, height, width), device=d()) # sample from the standard Gaussian

                        z = z * mix_latent + zp * (1. - mix_latent)

                x = mod(torch.cat([x, h, z], dim=1), time)
            else:
                x = mod(x)

        h = hs.pop()
        # z = zs.pop() if x0 is not None else torch.zeros_like(h)
        x = torch.cat([x, h], dim=1)
        # -- The final pop from `hs` is the input image.
        # -- Most UNets don't have the input on a residual, but this seems like an oversight to me.

        if x0 is None:
            return self.final(x)
        return self.final(x), kl_losses


class VAE(nn.Module):

    """
    VAE-only version of the Unet.
    """

    def __init__(self,
            res,
            channels = (8, 16, 32), # Number of channels at each level of the UNet
            num_blocks = 3,         # Number of res blocks per level
            mid_layers = 3,         # Number of linear layers in the middle block
        ):
        super().__init__()

        self.channels = channels
        self.num_blocks = num_blocks

        # Initial convolution up to the first res block
        self.initial = nn.Conv2d(3, channels[0], kernel_size=1, padding=0)

        self.encoder = nn.ModuleList()
        # -- This is analogous to the vcencoder in the VCUNet

        for i, ch in enumerate(channels):
            # Add a sequence of ResBlocks

            self.encoder.extend(ResBlockNT(in_channels=ch, out_channels=ch, vout=True)
                                  for _ in range(num_blocks))

            # Downsample
            self.encoder.append(nn.AvgPool2d(kernel_size=2))

            if i < len(channels) - 1:
                # Project up to next number of channels
                self.encoder.append(nn.Conv2d(ch, channels[i+1], kernel_size=1, padding=0))

        m = 2 ** len(channels)
        self.mres = res[0] // m, res[1] // m
        self.h = h = channels[-1] * self.mres[0] * self.mres[1]

        print(' -- vae unet: midblock hidden dim:', self.h)

        midblock_enc, midblock_dec = [], []

        for i in range(mid_layers):
            midblock_enc.append(nn.Linear(h, h))
            midblock_dec.append(nn.Linear(h, h))
            if i < mid_layers - 1:
                midblock_enc.append(nn.ReLU())
                midblock_dec.append(nn.ReLU())

        self.midblock_enc = nn.Sequential(*midblock_enc)
        self.midblock_dec = nn.Sequential(*midblock_dec)

        self.toz = nn.Linear(h, h*2) # Project to the sample for the middle latent

        rchannels = channels[::-1]
        self.decoder = nn.ModuleList()
        for i, ch in enumerate(rchannels):

            # Upsample
            self.decoder.append(
                Lambda(lambda x : F.interpolate(x, scale_factor=2, mode='nearest'))
            )

            # Add a sequence of ResBlocks
            self.decoder.extend(ResBlockNT(in_channels=ch * 2, out_channels=ch)
                                for _ in range(num_blocks))
            # -- In channels: `ch` from the previous block, `ch` from the
            #    vc connection

            if i < len(channels) - 1:
                # Project down to next number of channels
                self.decoder.append(nn.Conv2d(ch, rchannels[i+1], kernel_size=1, padding=0))

        # Final convolution down to the required number of output channels
        self.final = nn.Conv2d(channels[0], 3, kernel_size=1, padding=0)

    def forward(self, x=None, num=None):
        """
        :param x1: The input to the network. The image corrupted to t1.
        :param x0: the target of the network. The image corrupted to t0 (that is, to a lesser degree
           than x1). If None, the network will run in sampling mode, skipping the variational encoder
           branch. If given, the image will be encoded into a Gaussian distribution on the latent space.
        :return: If x0 is None, a (diff) prediction for x0. If x0 is given, a pair consisting of a (diff) prediction for
           x0 and the kl losses for the latent distributions.
        """

        run_enc = x is not None

        # Encoder branch
        if run_enc:

            b, c, h, w = x.size()

            zs = []
            kl_losses = []

            x = self.initial(x)

            for mod in self.encoder:
                if type(mod) == ResBlockNT:

                    x, z = mod(x) # -- The second output is the mean/var on the latent space.
                    zs.append(z) # encoded target output

                else:
                    x = mod(x)

            x = x.reshape(b, -1) # flatten

            x = self.midblock_enc(x) + x
            z = self.toz(x)

            c = z.size(1) // 2; assert c == self.h

            kl_losses.append(kl_loss(z[:, :c], z[:, c:]))
            z = sample(z[:, :c], z[:, c:])

        else: # Sample the middle latent
            b = num
            z = torch.randn(b, self.h)

        # Decoder branch
        x = self.midblock_dec(z) + z

        x = x.reshape(b, -1, *self.mres)

        for mod in self.decoder:

            if type(mod) == ResBlockNT:
                b, c, height, width = x.size()

                if not run_enc: # sample
                    z = torch.randn(size=(b, c, height, width), device=d())
                else:
                    z = zs.pop() # The latent from the encoder

                    assert z.size(1) == 2 * c

                    kl_losses.append(kl_loss(z[:, :c, :, :], z[:, c:, :, :]))
                    z = sample(z[:, :c, :, :], z[:, c:, :, :])

                x = mod(torch.cat([x, z], dim=1))
            else:
                x = mod(x)

        if not run_enc:
            return self.final(x)
        return self.final(x), kl_losses
