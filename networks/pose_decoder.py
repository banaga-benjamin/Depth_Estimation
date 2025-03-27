import torch
from torch import nn
from torch.nn import init
from torchinfo import summary
from torch.nn import functional


class PoseDecoder(nn.Module):
    def __init__(self, in_channels: int = 512, device: str = 'cpu'):
        super( ).__init__( )
        
        self.conv_elu = list( )
        self.conv_elu.append(nn.Conv2d(in_channels, in_channels // 2, kernel_size = (1,1), stride = (1, 1)))
        self.conv_elu.append(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.conv_elu.append(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
    
        self.conv = nn.Conv2d(in_channels // 2, 6, kernel_size = (1, 1), stride = (1, 1))
    
        self.conv.to(device)
        for conv_elu in self.conv_elu: conv_elu.to(device)

    def forward(self, input):
        conv_output = input.clone( )

        # pass input through convolution - elu layers
        for conv_elu in self.conv_elu:
            conv_output = conv_elu(conv_output)
            conv_output = functional.elu_(conv_output)
        conv_output = self.conv(conv_output)
        
        # get mean of features at each channel
        output = torch.zeros(conv_output.size(dim = 0), conv_output.size(dim = 1))
        for batch in range(conv_output.size(dim = 0)):
            for channel in range(conv_output.size(dim = 1)):
                output[batch][channel] = torch.mean(conv_output[batch][channel])
        return output


# for debugging
# if __name__ == "__main__":
#    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'

#    H = 192; W = 640
#    input = torch.rand(1, 512, H // 32, W // 32, device = device)
   
   
#    model = PoseDecoder(device = device); outputs = model(input)
#    for output in outputs: print(output)
