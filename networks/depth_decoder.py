import torch
from torch import nn
from torch.nn import init
from torch.nn import functional


class DepthDecoder(nn.Module):
    def __init__(self):
        super( ).__init__( )

        # initialize batchnorm, upscale, refinement, and compression layers
        self.bns = nn.ModuleList( )
        self.ups = nn.ModuleList( )
        self.refs = nn.ModuleList( )
        self.comps = nn.ModuleList( )
        for channels in [512, 256, 128, 64]:
            self.ups.append(nn.ModuleList( ))
            self.refs.append(nn.ModuleList( ))
            self.ups[-1].append(nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2)))
            self.refs[-1].append(nn.Conv2d(channels, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))

            if channels == 64: continue
            self.bns.append(nn.BatchNorm2d(num_features = channels // 2))
            self.ups[-1].append(nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size = (2, 2), stride = (2, 2)))
            self.comps.append(nn.Conv2d(channels // 2, channels // 4, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            self.refs[-1].append(nn.Conv2d(channels // 2, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        
        for ups in self.ups:
            for up in ups: init.kaiming_normal_(up.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(up.bias)
        
        for refs in self.refs:
            for ref in refs: init.kaiming_normal_(ref.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(ref.bias)
        
        for comp in self.comps: init.kaiming_normal_(comp.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(comp.bias)

        self.map_layer = (nn.Conv2d(32, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        init.kaiming_normal_(self.map_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.map_layer.bias)

        # initialize layers for combining results
        self.bn_comb = nn.BatchNorm2d(num_features = 2)

        self.refs_comb = nn.ModuleList( )
        self.refs_comb.append(nn.Conv2d(1, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.refs_comb.append(nn.Conv2d(1, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.refs_comb.append(nn.Conv2d(2, 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))

        for ref_comb in self.refs_comb:
            init.kaiming_normal_(ref_comb.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(ref_comb.bias)

        self.comp_comb = nn.Conv2d(2, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        init.kaiming_normal_(self.comp_comb.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.comp_comb.bias)


    def forward(self, input, candidate):
        # pass through convolutional layers to obtain temporary depth maps at scales
        outputs = list( )
        for idx, _ in enumerate([512, 256, 128, 64]):
            outputs.append(functional.relu(self.ups[idx][0](self.refs[idx][0](input[idx]) + input[idx])))

        # progressively combine depth maps at scales until final scale
        for idx, _ in enumerate([256, 128, 64]):
            temp = self.bns[idx](torch.cat([functional.relu(self.ups[idx][1](outputs[idx])), outputs[idx + 1]], dim = 1))
            outputs[idx + 1] = functional.relu(self.comps[idx](self.refs[idx][1](temp) + temp))
        output = functional.relu(self.map_layer(outputs[-1]))

        # combine depth map at final scale with candidate depth map
        temp = self.bn_comb(torch.cat([self.refs_comb[0](output), self.refs_comb[1](candidate)], dim = 1))
        final_output = functional.relu(self.comp_comb(self.refs_comb[2](temp) + temp))
        return final_output
