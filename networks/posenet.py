import torch
from torch import nn
from torchvision import models


class PoseNet(nn.Module):
    """
        predicts a pose matrix encoded into a vector of size [6] using pretrained ResNet18 as backbone
    """
    
    def __init__(self):
        super( ).__init__( )

        # copy resnet weights
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights = weights)
        
        self.resnet.fc = nn.Identity( )                 # do nothing
        self.linear = nn.Linear(512 * 2, 6)             # linear layer


    def forward(self, img_0: torch.Tensor, img_1: torch.Tensor):
        # extract features from images
        feat_0 = self.resnet(img_0)
        feat_1 = self.resnet(img_1)

        # concatenate features and pass to linear
        feat_cat = torch.cat([feat_0, feat_1], dim = 1)
        return self.linear(feat_cat)
