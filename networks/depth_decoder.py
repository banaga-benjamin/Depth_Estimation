import torch
from torch import nn
from torch.nn import init
from torch.nn import functional


class DepthDecoder(nn.Module):
    def __init__(self, in_channels: int = 512):
        super( ).__init__( )
        self.cost_layers = nn.ModuleList( )

        self.upscale_layers = nn.ModuleList( )
        self.upscale_layers.append(nn.ModuleList( ))
        self.upscale_layers.append(nn.ModuleList( ))

        self.map_layers = nn.ModuleList( )
        self.compress_layers = nn.ModuleList( )
        self.compress_layers.append(nn.ModuleList( ))
        self.compress_layers.append(nn.ModuleList( ))

        self.scales = 4
        channels = in_channels
        for scale in range(self.scales):
            if scale != 3:
                self.cost_layers.append(nn.Conv2d(channels // 4, channels // 2, kernel_size = (2, 2), stride = (2, 2)))
                init.kaiming_normal_(self.cost_layers[-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.cost_layers[-1].bias)

            self.upscale_layers[0].append(nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2)))
            self.upscale_layers[1].append(nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size = (2, 2), stride = (2, 2)))

            init.kaiming_normal_(self.upscale_layers[0][-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.upscale_layers[0][-1].bias)
            init.kaiming_normal_(self.upscale_layers[1][-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.upscale_layers[1][-1].bias)

            if scale != 0:
                self.compress_layers[0].append((nn.Conv2d(channels * 2, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))))
                init.kaiming_normal_(self.compress_layers[0][-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.compress_layers[0][-1].bias)
            self.compress_layers[1].append(nn.Conv2d(channels, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            init.kaiming_normal_(self.compress_layers[1][-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.compress_layers[1][-1].bias)

            self.map_layers.append(nn.Conv2d(channels // 4, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            init.kaiming_normal_(self.map_layers[-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.map_layers[-1].bias)

            channels //= 2
        self.cost_layers.append(nn.Conv2d(channels * 2, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        init.kaiming_normal_(self.cost_layers[-1].weight, mode = "fan_in", nonlinearity = "relu"); init.zeros_(self.cost_layers[-1].bias)
        self.cost_layers = nn.ModuleList(reversed(self.cost_layers))

        
    def forward(self, input, cost_volumes):
        # input cost should be of dimension (N, C, H, W)
        if cost_volumes.dim( ) < 4: cost_volumes = torch.unsqueeze(cost_volumes, dim = 0)
        
        # pass cost volumes through cost layers
        output = cost_volumes.clone( )
        cost_volumes = list( )
        for scale in range(self.scales):
            output = self.cost_layers[scale](output)
            cost_volumes.append(output)
        cost_volumes.reverse( ); del output

            
        outputs = list( )
        prev_scale_feature = None
        for scale in range(self.scales):
            # inputs should be of dimension (N, C, H, W)
            if input[scale].dim( ) < 4: input[scale] = torch.stack([input[scale]] * len(cost_volumes[scale]), dim = 0)
            elif len(input[scale]) < len(cost_volumes[scale]): input[scale] = torch.cat([input[scale]] * len(cost_volumes[scale]), dim = 0)

            if scale != 0:
                # concatenate input at current scale with previous scale feature
                output = torch.cat((input[scale], prev_scale_feature), dim = 1); del prev_scale_feature

                # compress concatenated features and pass to ELU
                output = functional.elu(self.compress_layers[0][scale - 1](output))

                # upscale feature and pass to ELU
                output = functional.elu(self.upscale_layers[0][scale](output))
            else:
                # upscale input at current scale and pass to ELU
                output = functional.elu(self.upscale_layers[0][scale](input[scale]))

            # concatenate with cost at current scale along C dimension
            output = torch.cat((output, cost_volumes[scale]), dim = 1)

            # compress concatenated features and pass to ELU
            output = functional.elu(self.compress_layers[1][scale](output))
            prev_scale_feature = output.clone( ).detach( )

            # upscale feature and pass to ELU
            output = functional.elu(self.upscale_layers[1][scale](output))

            # map feature to a single channel
            output = self.map_layers[scale](output)

            outputs.append(output)
        return outputs
