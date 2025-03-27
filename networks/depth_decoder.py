import torch
from torch import nn
from torch.nn import init
from torchinfo import summary
from torch.nn import functional


class DepthDecoder(nn.Module):
    def __init__(self, in_channels: int = 512, device: str = 'cpu'):
        super( ).__init__( )
        self.upscale_layers = list( )
        self.upscale_layers.append(list( ))
        self.upscale_layers.append(list( ))

        self.map_layers = list( )
        self.compress_layers = list( )
        self.compress_layers.append(list( ))
        self.compress_layers.append(list( ))

        self.scales = 4
        channels = in_channels
        for scale in range(self.scales):
            self.upscale_layers[0].append(nn.ConvTranspose2d(channels, channels // 2, kernel_size = (2, 2), stride = (2, 2)))
            self.upscale_layers[1].append(nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size = (2, 2), stride = (2, 2)))

            init.orthogonal_(self.upscale_layers[0][-1].weight); init.constant_(self.upscale_layers[0][-1].bias, 0.0)
            init.orthogonal_(self.upscale_layers[1][-1].weight); init.constant_(self.upscale_layers[1][-1].bias, 0.0)

            if scale != 0:
                self.compress_layers[0].append((nn.Conv2d(channels * 2, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))))
                init.orthogonal_(self.compress_layers[0][-1].weight); init.constant_(self.compress_layers[0][-1].bias, 0.0)
            self.compress_layers[1].append(nn.Conv2d(channels - channels // 4, channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            init.orthogonal_(self.compress_layers[1][-1].weight); init.constant_(self.compress_layers[1][-1].bias, 0.0)

            self.map_layers.append(nn.Conv2d(channels // 4, 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            init.orthogonal_(self.map_layers[-1].weight); init.constant_(self.map_layers[-1].bias, 0.0)
            
            channels //= 2

        for map_layer in self.map_layers: map_layer.to(device)
        for upscale_layer in self.upscale_layers[0]: upscale_layer.to(device)
        for upscale_layer in self.upscale_layers[1]: upscale_layer.to(device)
        for compress_layer in self.compress_layers[0]: compress_layer.to(device)
        for compress_layer in self.compress_layers[1]: compress_layer.to(device)


    def forward(self, input, cost):
        outputs = list( )
        prev_scale_feature = None
        for scale in range(self.scales):
            # inputs should be of dimension (N, C, H, W)
            if input[scale].dim( ) < 4: input[scale] = torch.unsqueeze(input[scale], dim = 0)
            if cost[scale].dim( ) < 4: cost[scale] = torch.unsqueeze(cost[scale], dim = 0)

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
            output = torch.cat((output, cost[scale]), dim = 1)

            # compress concatenated features and pass to ELU
            output = functional.elu(self.compress_layers[1][scale](output))
            prev_scale_feature = output.clone( ).detach( )

            # upscale feature and pass to ELU
            output = functional.elu(self.upscale_layers[1][scale](output))

            # map feature to a single channel
            output = self.map_layers[scale](output)

            outputs.append(output)
        return outputs

# for debugging
# if __name__ == "__main__":
#    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'

#    H = 192; W = 640
#    input = [torch.rand(1, 512, H // 32, W // 32, device = device), torch.rand(1, 256, H // 16, W // 16, device = device),
#             torch.rand(1, 128, H // 8, W // 8, device = device), torch.rand(1, 64, H // 4, W // 4, device = device)]
   
#    cost = [torch.rand(1, 256, H // 16, W // 16, device = device), torch.rand(1, 128, H // 8, W // 8, device = device),
#            torch.rand(1, 64, H // 4, W // 4, device = device), torch.rand(1, 32, H // 2, W // 2, device = device)]
   
#    model = DepthDecoder(device = device); outputs = model(input, cost)
#    for output in outputs: print(output.shape)
