import torch
from torch import nn
from torch.nn import init
from torch.nn import functional


class DepthDecoder(nn.Module):
    def __init__(self):
        super( ).__init__( )

        # initialize batchnorm layers
        self.batchnorms = nn.ModuleList( )
        self.batchnorms.append(nn.BatchNorm2d(num_features = 128))
        self.batchnorms.append(nn.BatchNorm2d(num_features = 2))
        
        # initialize upsampling blocks
        self.conv_layers = nn.ModuleList( )
        self.upscale_layers = nn.ModuleList( )
        for channels in (512, 256, 128, 64):
            self.conv_layers.append(nn.Conv2d(channels, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            self.upscale_layers.append(nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2)))

            init.kaiming_normal_(self.conv_layers[-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.conv_layers[-1].bias)
            init.kaiming_normal_(self.upscale_layers[-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.upscale_layers[-1].bias)

        # initialize refinement layers
        self.ref_layers = nn.ModuleList( )
        self.ref_layers.append(nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.ref_layers.append(nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.ref_layers.append(nn.Conv2d(2, 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))

        init.kaiming_normal_(self.ref_layers[0].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.ref_layers[0].bias)
        init.kaiming_normal_(self.ref_layers[1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.ref_layers[1].bias)
        init.kaiming_normal_(self.ref_layers[2].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.ref_layers[2].bias)

        # initialize map/reduction layers
        self.map_layers = nn.ModuleList( )
        self.map_layers.append(nn.Conv2d(32, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.map_layers.append(nn.Conv2d(2, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))

        init.kaiming_normal_(self.map_layers[0].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.map_layers[0].bias)
        init.kaiming_normal_(self.map_layers[1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.map_layers[1].bias)

                    
    def forward(self, input, candidate):
        output = input
        for idx, channels in enumerate([512, 256, 128, 64]):   # sequentially upsample input
            output = functional.relu(self.upscale_layers[idx](functional.relu(self.conv_layers[idx](output)) + output))
            if channels == 256: output = self.batchnorms[0](output)     # batch norm
        
        # map input to a single dimension, and refine input and candidate
        output = functional.relu(self.ref_layers[0](functional.relu(self.map_layers[0](output))))
        candidate = functional.relu(self.ref_layers[1](candidate))

        # concatenate refined input and candidate and normalize
        cat_feat = self.batchnorms[1](torch.cat([output, candidate], dim = 1))

        # refine concatenated features and map to a single dimension
        output = functional.relu(self.map_layers[1](functional.relu(self.ref_layers[2](cat_feat)) + cat_feat))
        return output
