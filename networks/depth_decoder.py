import torch
import numpy as np
from torch import nn
from torch.nn import init
from torch.nn import functional


class DepthDecoder(nn.Module):
    def __init__(self, in_channels: int = 512, cost_channels: int = 512):
        super( ).__init__( )
        self.in_channels = in_channels
        self.cost_channels = cost_channels

        channels = in_channels
        self.upscale_layers = nn.ModuleList( )
        self.cost_compress_layers = nn.ModuleList( )
        self.upscale_layers.append(nn.ModuleList( ))
        for _ in range(int(np.log2(channels)) - int(np.log2(64))):
            self.upscale_layers[0].append(nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2)))
            init.kaiming_normal_(self.upscale_layers[0][-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.upscale_layers[0][-1].bias)

            channels //= 2

        channels = cost_channels
        for _ in range(int(np.log2(channels)) - int(np.log(64))):
            self.cost_compress_layers.append(nn.Conv2d(channels, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            init.kaiming_normal_(self.cost_compress_layers[-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.cost_compress_layers[-1].bias)

            channels //= 2

        channels = 64
        self.upscale_layers.append(nn.ModuleList( ))
        for _ in range(int(np.log2(64)) - int(np.log2(16))):
            self.upscale_layers[1].append(nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2)))
            init.kaiming_normal_(self.upscale_layers[1][-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.upscale_layers[1][-1].bias)

            channels //= 2

        self.compress_layer = nn.Conv2d(128, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        init.kaiming_normal_(self.compress_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.compress_layer.bias)

        self.map_layer = nn.Conv2d(16, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        init.kaiming_normal_(self.map_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.map_layer.bias)
    

    def forward(self, input, cost_volume):
        # inputs should be in (N, C, H, W) format
        if cost_volume.dim( ) < 4: cost_volume = cost_volume.unsqueeze( )
        if input.dim( ) < 4: input = torch.stack([input] * cost_volume.size(dim = 0))
        elif input.size(dim = 0) < cost_volume.size(dim = 0): input = torch.cat([input] * cost_volume.size(dim = 0), dim = 0)

        output = input
        # progressively upscale input until (N, 64, H, W)
        # progressively compress cost volume until (N, 64, H, W)
        for idx in range(int(np.log2(self.in_channels)) - int(np.log2(64))):
            output = functional.elu_(self.upscale_layers[0][idx](output))
        
        for idx in range(int(np.log2(self.cost_channels)) - int(np.log2(64))):
            cost_volume = functional.elu_(self.cost_compress_layers[idx](cost_volume))

        # combine cost volume and input to (N, 128, H, W) and compress to (N, 64, H, W)
        output = functional.elu_(self.compress_layer(torch.cat((output, cost_volume), dim = 1)))

        # progressively upscale from (N, 64, H, W) to (N, 16, H', W')
        for idx in range(int(np.log2(64)) - int(np.log2(16))):
            output = functional.elu_(self.upscale_layers[1][idx](output))

        # map to (N, 1, H', W') and normalize
        return functional.sigmoid(self.map_layer(output))
