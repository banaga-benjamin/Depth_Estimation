import torch
from torch import nn
from torch.nn import init
from torch.nn import functional


class DepthDecoder(nn.Module):
    def __init__(self):
        super( ).__init__( )

        # initialize basic block layers
        self.basic_blocks = nn.ModuleList( )
        for channels in [512, 256, 128, 64]:
            self.basic_blocks.append(BasicBlock(channels))
        
        # initialize aggregation blocks
        self.aggregation_blocks = nn.ModuleList( )
        for channels in [256, 128, 64]:
            self.aggregation_blocks.append(AggregationBlock(channels))

        # initialize map layer
        self.map_layer = nn.Conv2d(32, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        init.kaiming_normal_(self.map_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.map_layer.bias)
        
        # initialize combination block
        self.combination_blocks = CombinationBlock( )


    def forward(self, input: torch.Tensor, candidate: torch.Tensor):
        # pass inputs through basic blocks
        outputs = list( )
        for idx, _ in enumerate([512, 256, 128, 64]):
            outputs.append(self.basic_blocks[idx](input[idx]))
        
        # aggregate basic block results
        for idx, _ in enumerate([256, 128, 64]):
            outputs[idx + 1] = self.aggregation_blocks[idx](outputs[idx], outputs[idx + 1])
        
        # map final aggregation result to a single channel
        outputs[-1] = functional.relu(self.map_layer(outputs[-1]))

        # obtain final result by combining final aggregation result with candidate
        final_result = self.combination_blocks(outputs[-1], candidate)
        return final_result



class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super( ).__init__( )

        # initialize layers for refining and upscaling results
        self.ups_layer = nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2))
        self.ref_layer = nn.Conv2d(channels, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        init.kaiming_normal_(self.ups_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.ups_layer.bias)
        init.kaiming_normal_(self.ref_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.ref_layer.bias)


    def forward(self, input: torch.Tensor):
        return functional.relu(self.ups_layer(self.ref_layer(input) + input))



class AggregationBlock(nn.Module):
    def __init__(self, channels: int):
        super( ).__init__( )

        # initialize layers for aggregating results
        self.batchnorm = nn.BatchNorm2d(num_features = channels)
        self.ups_layer = nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2))
        self.ref_layer = nn.Conv2d(channels, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.comp_layer = nn.Conv2d(channels, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        init.kaiming_normal_(self.ups_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.ups_layer.bias)
        init.kaiming_normal_(self.ref_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.ref_layer.bias)
        init.kaiming_normal_(self.comp_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.comp_layer.bias)


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        temp = self.batchnorm(torch.cat([functional.relu(self.ups_layer(input_0)), input_1], dim = 1))
        return functional.relu(self.comp_layer(self.ref_layer(temp) + temp))



class CombinationBlock(nn.Module):
    def __init__(self):
        super( ).__init__( )

        # initialize layers for combining results
        self.ref_layers = nn.ModuleList( )
        self.batchnorm = nn.BatchNorm2d(num_features = 2)
        self.comp_layer = nn.Conv2d(2, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.ref_layers.append(nn.Conv2d(1, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.ref_layers.append(nn.Conv2d(1, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.ref_layers.append(nn.Conv2d(2, 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))


        init.kaiming_normal_(self.comp_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.comp_layer.bias)
        for ref_layer in self.ref_layers: init.kaiming_normal_(ref_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(ref_layer.bias)

    
    def forward(self, input: torch.Tensor, candidate: torch.Tensor):
        temp = self.batchnorm(torch.cat([functional.relu(self.ref_layers[0](input)), functional.relu(self.ref_layers[1](candidate))], dim = 1))
        return functional.relu(self.comp_layer(self.ref_layers[2](temp) + temp))
