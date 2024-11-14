import torch
from torch import nn
import torch.nn.functional as F

from .tools import coords, d, Lambda

# # Number of groups in the group normalization
# NUM_GROUPS = 32

class ResBlock(nn.Module):
    """
    Simplified residual block. Applies a convolution, and a residual connection around it. Fixed number of channels.

    """

    def __init__(self, channels, dropout=0.1, double_in=False, time_emb=512):
        """

        :param channels:
        :param dropout:
        :param double_in: Double the number of input channels. This is required for a UNet where the skip connections
            are concatenated rather than added.
        """

        super().__init__()
        self.double_in = double_in

        self.in_channels = channels * 2 if double_in else channels
        self.resconv = nn.Conv2d(self.in_channels, channels, 1, padding=0) if double_in else None

        self.convolution = nn.Sequential(
            nn.GroupNorm(1, self.in_channels), # Equivalent to LayerNorm, but over the channel dimension of an image
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.in_channels, channels, 3, padding=1)
        )

        # Projects the time scalars up to the number of input channels
        self.embed_time = nn.Linear(time_emb, self.in_channels)

        # Projects the pixel coordinates up to the number of channels
        self.embed_coords = nn.Conv2d(2, self.in_channels, 1, padding=0)

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
        res = self.resconv(x) if self.double_in else x
        return self.convolution(x + time + crds) + res

class UNet(nn.Module):
    def __init__(self,
            res,
            channels = (8, 16, 32), # Number of channels at each level of the UNet
            num_blocks = 3,         # Number of res blocks per level
            mid_layers = 3,         # Number of linear layers in the middle block
            res_cat=False,
            max_time=None,          # Maximum number of timesteps
            time_emb=512,           # Time embedding dimension
            disc_time=False         # If true, time is discrete (positive ints) else, continuous in (0, 1)
        ):
        super().__init__()

        self.channels = channels
        self.num_blocks = num_blocks
        self.res_cat = res_cat

        # Initial convolution up to the first res block
        self.initial = nn.Conv2d(3, channels[0], kernel_size=1, padding=0)

        self.encoder = nn.ModuleList()
        for i, ch in enumerate(channels):
            # Add a sequence of ResBlocks
            self.encoder.extend(ResBlock(ch) for _ in range(num_blocks))

            # Downsample
            self.encoder.append(nn.AvgPool2d(kernel_size=2))

            if i < len(channels) - 1:
                # Project up to next number of channels
                self.encoder.append(nn.Conv2d(ch, channels[i+1], kernel_size=1, padding=0))

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
            self.decoder.extend(ResBlock(ch, double_in=res_cat) for _ in range(num_blocks))

            if i < len(channels) - 1:
                # Project down to next number of channels
                self.decoder.append(nn.Conv2d(ch, rchannels[i+1], kernel_size=1, padding=0))

        # Final convolution down to the required number of output channels
        self.final = nn.Conv2d(channels[0] + 3, 3, kernel_size=1, padding=0)

        self.disc_time = disc_time
        if disc_time:
            # Discrete time, use embedding vectors
            self.timeembs = nn.Embedding(num_embeddings=max_time, embedding_dim=time_emb)
        else:
            # Continuous time (0, 1), use a nonlinear function to project up
            self.timeembs = nn.Sequential(
                nn.Linear(1, time_emb * 2), nn.ReLU(),
                nn.Linear(time_emb * 2, time_emb * 2), nn.ReLU(),
                nn.Linear(time_emb * 2, time_emb),
            )

    def forward(self, x, time):

        b, c, h, w = x.size()

        if type(time) is float or type(time) is int:
            time = torch.tensor([time], device=d())

        assert len(time.size()) == 1 and (time.size(0) == 1 or time.size(0) == b), str(time.size())

        if time.size(0) == 1:
            time = time.expand(b, 1)

        time = self.timeembs(time) # This is either an embedding layer or a feedforward network

        hs = [x] # Collect values for skip connections, start with the input image

        x = self.initial(x) # Project up to the first nr. of channels

        # Encoder branch
        for mod in self.encoder:
            if type(mod) == ResBlock:
                x = mod(x, time)
                hs.append(x)
            else:
                x = mod(x)

        # Mid blocks
        x = x.reshape(b, -1) # flatten

        x = self.midblock(x) + x
        x = x.reshape(b, -1, *self.mres)

        # Decoder branch
        for mod in self.decoder:
            if type(mod) == ResBlock:
                h = hs.pop() # The value from the relevant skip connection
                x = mod(torch.cat([x, h], dim=1), time) if self.res_cat \
                    else mod(x + h, time)
            else:
                x = mod(x)

        x = torch.cat([x, hs.pop()], dim=1) # -- The final pop from `hs` is the input image.
        return self.final(x)


#
# class UNetOld(nn.Module):
#     """
#     A fairly arbitrary UNet.
#     """
#
#     def __init__(self, a=16, b=32, c=128, ls=2, krnl=3, res=(64, 64), mlm_offset=0.0, ln_params=True, num_mids=3,
#                  numouts=6):
#         super().__init__()
#
#         self.latent_size = ls
#         self.mlm_offset = mlm_offset
#
#         self.res = res
#         self.mr = mr = res[0] // 2**3, res[1] // 2**3
#
#         pm = 'zeros'
#         # non-linearity
#         self.nl = F.relu
#         # self.nl = lambda x : torch.sigmoid(x * 1e3) # torch.sign, F.relu
#         # -- A sigmoid nonlinearity with a temperature parameter offers a nice way to tune between high and low frequency
#         #    structures
#
#         pad = krnl//2
#         krnl = (krnl, krnl)
#
#         self.coords0 = coords(res[0],      res[1])
#         self.coords1 = coords(res[0] // 2, res[1] // 2)
#         self.coords2 = coords(res[0] // 4, res[1] // 4)
#         self.coords3 = coords(res[0] // 8, res[1] // 8)
#
#         self.conve11 = nn.Conv2d(3+3, a, krnl, padding=pad, padding_mode=pm)
#         self.conve1point = nn.Conv2d(a+3, a, (1, 1), padding=0)
#
#         self.ln1 = nn.LayerNorm(a, elementwise_affine=ln_params)
#
#         self.conve21 = nn.Conv2d(a+3, b, krnl, padding=pad, padding_mode=pm)
#         self.conve22 = nn.Conv2d(b+3, b, krnl, padding=pad, padding_mode=pm)
#         self.conve2point = nn.Conv2d(b+3, b, (1, 1), padding=0)
#
#         self.ln2 = nn.LayerNorm(b, elementwise_affine=ln_params)
#
#         self.conve31 = nn.Conv2d(b+3, c, krnl, padding=pad, padding_mode=pm)
#         self.conve32 = nn.Conv2d(c+3, c, krnl, padding=pad, padding_mode=pm)
#         self.conve3point = nn.Conv2d(c+3, c, (1, 1), padding=0)
#
#         self.ln3 = nn.LayerNorm(c, elementwise_affine=ln_params)
#
#         self.line = nn.Linear(c * mr[0] * mr[1], ls)
#
#         mids = [nn.Linear(ls, ls) for _ in range(num_mids)]
#         self.mids = nn.ModuleList(mids)
#
#         self.lind = nn.Linear(ls, c * mr[0] * mr[1])
#
#         self.convd3point = nn.Conv2d(c+3, c, (1, 1), padding=0)
#
#         self.ln4 = nn.LayerNorm(c, elementwise_affine=ln_params)
#
#         self.convd32 = nn.ConvTranspose2d(c+3, c, krnl, padding=pad)
#         self.convd31 = nn.ConvTranspose2d(c+3, b, krnl, padding=pad)
#
#         self.convd2point = nn.Conv2d(b+3, b, (1, 1), padding=0)
#
#         self.ln5 = nn.LayerNorm(b, elementwise_affine=ln_params)
#
#         self.convd22 =nn.ConvTranspose2d(b+3, b, krnl, padding=pad)
#         self.convd21 =nn.ConvTranspose2d(b+3, a, krnl, padding=pad)
#
#         self.convd1point = nn.Conv2d(a+3, a, (1, 1), padding=0)
#
#         self.ln6 = nn.LayerNorm(a, elementwise_affine=ln_params)
#
#         self.convd11 = nn.ConvTranspose2d(a+3, numouts, krnl, padding=pad)
#         # -- We have six output channels. 3 for the means, 3 for the variances.
#
#         self.alphas = torch.tensor([1., 1., 1.])
#         self.betas  = torch.tensor([1., 1., 1.])
#
#         # -- We have four output channels. The fourth is a probability distribution that tells us how the input and
#         #    and output are mixed in the sample.
#
#     def coordconv(self, img, coords, step):
#
#         b, c, h, w = img.size()
#         assert coords.size() == (1, 2, h, w)
#
#         step = torch.full(fill_value=step, size=(b, 1, h, w), device=d())
#
#         return torch.cat([img, coords.expand(b, 2, h, w), step], dim=1)
#
#     def forward(self, img, step):
#
#         b, c, h, w = img.size()
#
#         # forward pass
#         x = img
#
#         x = self.coordconv(x, self.coords0, step)
#         x = self.nl(self.conve11(x))
#
#         x = self.coordconv(x, self.coords0, step)
#         x = x_e11 = F.relu(self.conve1point(x))
#
#         x = F.max_pool2d(x, kernel_size=2)
#         x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         x = self.coordconv(x, self.coords1, step)
#         x = self.nl(self.conve21(x))
#
#         x = self.coordconv(x, self.coords1, step)
#         x = self.nl(self.conve22(x))
#
#         x = self.coordconv(x, self.coords1, step)
#         x = x_e22 = self.nl(self.conve2point(x))
#
#         x = F.max_pool2d(x, kernel_size=2)
#         x = self.nl(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         x = self.coordconv(x, self.coords2, step)
#         x = self.nl(self.conve31(x))
#
#         x = self.coordconv(x, self.coords2, step)
#         x = self.nl(self.conve32(x))
#
#         x = self.coordconv(x, self.coords2, step)
#         x = x_e32 = self.nl(self.conve3point(x))
#
#         x = F.max_pool2d(x, kernel_size=2)
#         x = self.ln3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         x = self.line(x.reshape(b, -1))
#
#         # Middle layers of the unet
#         for mid in self.mids:
#             x = self.nl(mid(x))
#
#         x = self.lind(x).reshape(b, -1, *self.mr)
#
#         x = F.upsample_bilinear(x, scale_factor=2) # --
#         x = self.ln4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         x = self.coordconv(x, self.coords2, step)    # point
#         x = self.nl(self.convd3point(x))
#
#         x = x * self.betas[0] + x_e32 * self.alphas[0] # residual
#
#         x = self.coordconv(x, self.coords2, step)
#         x = self.nl(self.convd32(x))
#
#         x = self.coordconv(x, self.coords2, step)
#         x = self.nl(self.convd31(x))
#
#         x = F.upsample_bilinear(x, scale_factor=2) # --
#         x = self.ln5(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         x = self.coordconv(x, self.coords1, step)    # point
#         x = self.nl(self.convd2point(x))
#
#         x = x * self.betas[1] + x_e22 * self.alphas[1] # res
#
#         x = self.coordconv(x, self.coords1, step)
#         x = self.nl(self.convd22(x))
#
#         x = self.coordconv(x, self.coords1, step)
#         x = self.nl(self.convd21(x))
#
#         x = F.upsample_bilinear(x, scale_factor=2) # --
#
#         x = self.coordconv(x, self.coords0, step)    # point
#         x = self.nl(self.convd1point(x))
#
#         x = self.ln6(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         x = x * self.betas[2] + x_e11 * self.alphas[2]
#
#         x = self.coordconv(x, self.coords0, step)
#         return self.convd11(x)
#
