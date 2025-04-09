from torch import nn
from torch import unsqueeze
from torchvision import models


class DepthEncoder(nn.Module):
    """
        extracts multi-scale features from an input image using pretrained ResNet18
    """
    
    def __init__(self):
        super( ).__init__( )

        # copy resnet weights
        resnet_weights = models.ResNet18_Weights.DEFAULT
        reference_model = models.resnet18(weights = resnet_weights)

        self.bn1 = reference_model.bn1
        self.relu = reference_model.relu
        self.conv1 = reference_model.conv1
        self.maxpool = reference_model.maxpool

        self.layer1 = reference_model.layer1
        self.layer2 = reference_model.layer2
        self.layer3 = reference_model.layer3
        self.layer4 = reference_model.layer4

        for parameter in self.parameters( ): parameter.requires_grad = False


    def forward(self, input):
        # input should be of dimension (N, C, H, W)
        if input.dim( ) < 4: input = unsqueeze(input, dim = 0)
        
        # preprocess input before passing to residual layers
        preprocess = self.maxpool(self.relu(self.bn1(self.conv1(input))))
    
        # obtain layer outputs
        outputs = list( )
        outputs.append(self.layer1(preprocess))
        outputs.append(self.layer2(outputs[-1]))
        outputs.append(self.layer3(outputs[-1]))
        outputs.append(self.layer4(outputs[-1]))
        outputs.reverse( )
        return outputs
