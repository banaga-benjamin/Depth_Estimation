import torch
from torch import nn
from torch.nn import init
from torch.nn import functional


class DepthDecoder(nn.Module):
    """
        predicts a depth map using multi-scale features and candidate depth map as inputs
    """

    def __init__(self, final_height: int = 192, final_width: int = 640):
        super( ).__init__( )

        # initialize basic block layers
        height = 6; width = 20
        self.basic_blocks = nn.ModuleList( )
        for channels in [512, 256, 128, 64]:
            height *= 2; width *= 2
            self.basic_blocks.append(BasicBlock(channels, height, width))

        # initialize upsampling blocks
        self.upsampling_blocks = nn.ModuleList( )
        for channels in (128, 64, 32):
            self.upsampling_blocks.append(UpscaleBlock(channels))

        # initialize layernorm layer
        self.layernorm = nn.LayerNorm([16 * 4, final_height, final_width])

        # initialize combination block
        self.combination_block = CombinationBlock(height = final_height, width = final_width)


    def forward(self, input: torch.Tensor, candidate: torch.Tensor):
        # pass inputs through basic blocks
        outputs = list( )
        for idx, _ in enumerate([512, 256, 128, 64]):
            outputs.append(self.basic_blocks[idx](input[idx]))

        # pass basic block outputs to upsampling blocks
        for idx, _ in enumerate([128, 64, 32]):
            outputs[idx] = self.upsampling_blocks[idx](outputs[idx])
        
        # stack outputs at different scales, normalize, and take average
        depth_output = torch.mean(self.layernorm(torch.cat(outputs, dim = 1)), dim = 1, keepdim = True)

        # combine depth result with candidate depth map
        final_output = self.combination_block(depth_output, candidate)

        return final_output



class BasicBlock(nn.Module):
    """
        predicts a depth map at given scale as specified by channels parameter using convolutional layers with residual connections
    """

    def __init__(self, channels: int, height: int, width: int):
        super( ).__init__( )

        # initialize layernorm layer
        self.layernorm  = nn.LayerNorm([channels // 2, height, width])

        # initialize refinement layers
        self.ref_layers = nn.ModuleList( )
        self.ref_layers.append(nn.Conv2d(channels, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.ref_layers.append(nn.Conv2d(channels // 2, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))

        for ref_layer in self.ref_layers: init.kaiming_normal_(ref_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(ref_layer.bias)

        # initialize upscaling layer
        self.ups_layers = nn.ModuleList( )
        for channels in [channels, channels // 2]:
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")

            conv = nn.Conv2d(channels, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
            init.kaiming_normal_(conv.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(conv.bias)

            self.ups_layers.append(nn.Sequential(upsample, conv))


    def forward(self, input: torch.Tensor):
        # refine, upsample. and normalize input
        output = self.layernorm(functional.relu(self.ups_layers[0](self.ref_layers[0](input) + input)))

        # further refine and upsample input
        return functional.relu(self.ups_layers[1](self.ref_layers[1](output) + output))



class UpscaleBlock(nn.Module):
    """
        upscales a depth map at a given scale as specified by channels parameter using convolutional layers and residual connections
    """

    def __init__(self, channels: int):
        super( ).__init__( )
    
        # initialize upsampling layers
        self.ups_layers = nn.ModuleList( )
        while channels > 16:
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")

            conv = nn.Conv2d(channels, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
            init.kaiming_normal_(conv.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(conv.bias)

            self.ups_layers.append(nn.Sequential(upsample, conv)); channels //= 2
        

    def forward(self, input: torch.Tensor):
        # pass output through upsampling layers
        output = input
        for ups_layer in self.ups_layers:
            output = functional.relu(ups_layer(output))
        return output



class CombinationBlock(nn.Module):
    """
        combines an input depth map and candidate depth map to obtain a final depth map using convolutional layers and residual connections
    """

    def __init__(self, height: int, width: int):
        super( ).__init__( )

        # initialize layers for combining results
        self.ref_layers = nn.ModuleList( )
        self.layernorm = nn.LayerNorm([2, height, width])
        self.comp_layer = nn.Conv2d(2, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.ref_layers.append(nn.Conv2d(1, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.ref_layers.append(nn.Conv2d(1, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.ref_layers.append(nn.Conv2d(2, 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))


        init.kaiming_normal_(self.comp_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.comp_layer.bias)
        for ref_layer in self.ref_layers: init.kaiming_normal_(ref_layer.weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(ref_layer.bias)

    
    def forward(self, input: torch.Tensor, candidate: torch.Tensor):
        # refine, combine, and normalize input and candidate
        output = self.layernorm(torch.cat([functional.relu(self.ref_layers[0](input)), functional.relu(self.ref_layers[1](candidate))], dim = 1))

        # refine combined results and compress to a single channel
        return functional.relu(self.comp_layer(self.ref_layers[2](output) + output))
