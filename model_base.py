from abc import ABC, abstractmethod
from dsketch.raster.composite import softor
from dsketch.raster.raster import exp, nearest_neighbour, compute_nearest_neighbour_sigma_bres
import torch
import torch.nn as nn
from dsketch.raster.composite import softor
from dsketch.raster.disttrans import catmull_rom_spline, curve_edt2_bruteforce, curve_edt2_polyline, line_edt2
from dsketch.raster.raster import exp

class Decoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode_to_params(self, inp):
        pass

    @abstractmethod
    def create_edt2(self, params):
        pass

    def raster_soft(self, edt2, sigma2):
        rasters = exp(edt2, sigma2)
        return softor(rasters, keepdim=True)

    def raster_hard(self, edt2):
        rasters = nearest_neighbour(edt2.detach(), compute_nearest_neighbour_sigma_bres(self.grid))
        return softor(rasters, keepdim=True)

    @abstractmethod
    def get_sigma2(self, params):
        pass


    def forward(self, inp):
        params = self.decode_to_params(inp)
        sigma2 = self.get_sigma2(params)
        edt2 = self.create_edt2(params)
        images = self.raster_soft(edt2, sigma2)

        return images

class AdjustableSigmaMixin:
    """
    Provides a sigma2 parameter that isn't learned and can be adjusted externally
    """

    def __init__(self):
        super().__init__()

    def get_sigma2(self, params=None):
        return self.sigma2

    def set_sigma2(self, value):
        self.sigma2 = value

class SinglePassSimpleLineDecoder(AdjustableSigmaMixin, Decoder):
    def __init__(self, nlines=5, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2):
        super().__init__()

        # build the coordinate grid:
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

        self.latent_to_linecoord = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, nlines * 4),
            nn.Tanh()
        )

        self.sigma2 = sigma2


    def decode_to_params(self, inp):
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = inp.shape[0]

        lines = self.latent_to_linecoord(inp)  # [batch, nlines*4]
        lines = lines.view(bs, -1, 2, 2)  # expand -> [batch, nlines, 2, 2]

        return lines

    def create_edt2(self, lines):
        edt2 = line_edt2(lines, self.grid)

        return edt2
    
