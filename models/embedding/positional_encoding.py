"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
import math
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute positional encoding with different periodic functions.
    Supported: 'sinusoid', 'triangular', 'square', 'sawtooth'
    """

    def __init__(self, d_model, max_len, device, periodic_func='sinusoid'):
        """
        constructor of positional encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        :param periodic_func: type of periodic function
                             ('sinusoid', 'triangular', 'square', 'sawtooth')
        """
        super(PositionalEncoding, self).__init__()
        self.periodic_func = periodic_func

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        # Compute the argument for periodic functions
        z_even = pos / (10000 ** (_2i / d_model))

        if d_model % 2 == 0:
            z_odd = z_even
        else:
            _2i_odd = torch.arange(0, d_model - 1, step=2, device=device).float()
            z_odd = pos / (10000 ** (_2i_odd / d_model))

        if periodic_func == 'sinusoid':
            # Original sinusoidal encoding
            self.encoding[:, 0::2] = torch.sin(z_even)
            self.encoding[:, 1::2] = torch.cos(z_odd)

        elif periodic_func == 'triangular':
            # Triangular wave: continuous piecewise-linear, period 2π
            self.encoding[:, 0::2] = self._triangular_wave(z_even)
            self.encoding[:, 1::2] = self._triangular_wave(z_odd)

        elif periodic_func == 'square':
            # Square wave: -1 in [0, π), +1 in [π, 2π)
            self.encoding[:, 0::2] = self._square_wave(z_even)
            self.encoding[:, 1::2] = self._square_wave(z_odd)

        elif periodic_func == 'sawtooth':
            # Sawtooth wave: ramp with period 2π
            self.encoding[:, 0::2] = self._sawtooth_wave(z_even)
            self.encoding[:, 1::2] = self._sawtooth_wave(z_odd)

        else:
            raise ValueError(
                f"Unknown periodic function: {periodic_func}. "
                f"Use 'sinusoid', 'triangular', 'square', or 'sawtooth'."
            )
    
    @staticmethod
    def _triangular_wave(z):
        """
        Triangular wave function: tri(z)
        Piecewise-linear, periodic with period 2π, range [-1, 1]

        Definition (extended periodically):
          tri(z) =  2z/π              for z in [0, π/2)
                  -2z/π + 2          for z in [π/2, 3π/2)
                   2z/π - 4          for z in [3π/2, 2π)
        """
        z_mod = torch.fmod(z, 2 * math.pi)
        # ensure in [0, 2π)
        z_mod = torch.where(z_mod < 0, z_mod + 2 * math.pi, z_mod)

        pi = math.pi
        out = torch.empty_like(z_mod)

        cond1 = (z_mod >= 0) & (z_mod < 0.5 * pi)
        cond2 = (z_mod >= 0.5 * pi) & (z_mod < 1.5 * pi)
        cond3 = (z_mod >= 1.5 * pi) & (z_mod < 2.0 * pi)

        out[cond1] = (2.0 * z_mod[cond1]) / pi
        out[cond2] = (-2.0 * z_mod[cond2]) / pi + 2.0
        out[cond3] = (2.0 * z_mod[cond3]) / pi - 4.0

        return out

    @staticmethod
    def _square_wave(z):
        """
        Square wave function: sqw(z)
        -1 in [0, π), +1 in [π, 2π), periodic with period 2π
        """
        z_mod = torch.fmod(z, 2 * math.pi)
        # ensure in [0, 2π)
        z_mod = torch.where(z_mod < 0, z_mod + 2 * math.pi, z_mod)
        return torch.where(
            z_mod < math.pi,
            torch.tensor(-1.0, device=z.device),
            torch.tensor(1.0, device=z.device),
        )

    @staticmethod
    def _sawtooth_wave(z):
        """
        Sawtooth wave function: saw(z)
        Linear ramp periodic with period 2π
        z in [0, π], (z - 2π) in [π, 2π]
        """
        z_mod = torch.fmod(z, 2 * math.pi)
        # ensure in [0, 2π)
        z_mod = torch.where(z_mod < 0, z_mod + 2 * math.pi, z_mod)
        return torch.where(z_mod <= math.pi, z_mod, z_mod - 2 * math.pi)

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
