import torch
from torch import nn
from torch.nn import init
from torchinfo import summary
from torch.nn import functional


class PoseDecoder(nn.Module):
    def __init__(self, in_channels: int = 1024, device: str = 'cpu'):
        super( ).__init__( )
        
        self.device = device
        self.conv_elu = nn.ModuleList( )
        self.conv = nn.Conv2d(in_channels // 2, 6, kernel_size = (1, 1), stride = (1, 1))
        self.conv_elu.append(nn.Conv2d(in_channels, in_channels // 2, kernel_size = (1,1), stride = (1, 1)))
        self.conv_elu.append(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.conv_elu.append(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
    

    def forward(self, target_feature, src_features):
        # target feature should have dimensions (1, C, H, W)
        # source features should have dimensions (N, C, H, W)
        if target_feature.dim( ) < 4: target_feature = target_feature.unsqueeze(dim = 0)
        target_features = torch.cat([target_feature] * src_features.size(dim = 0), dim = 0)
        
        input = torch.cat((target_features, src_features), dim = 1)
        
        # pass input through convolution - elu layers
        conv_output = input
        for conv_elu in self.conv_elu:
            conv_output = conv_elu(conv_output)
            conv_output = functional.elu_(conv_output)
        conv_output = self.conv(conv_output)
        
        # get mean of features at each channel
        output = torch.zeros(conv_output.size(dim = 0), conv_output.size(dim = 1), device = self.device)
        for batch in range(conv_output.size(dim = 0)):
            for channel in range(conv_output.size(dim = 1)):
                output[batch][channel] = torch.mean(conv_output[batch][channel])
        return output


# for debugging
# if __name__ == "__main__":
#    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'

#    H = 192; W = 640
#    input = torch.rand(1, 1024, H // 32, W // 32, device = device)
   
   
#    model = PoseDecoder(device = device, in_channels = 1024); outputs = model(input)
#    for output in outputs: print(output)
