import torch
from torch import nn
from torchinfo import summary
from torchvision import models


class PoseEncoder(nn.Module):
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
        # input dimension should be (N, C, H, W)
        if input.dim( ) < 4: input = torch.unsqueeze(input, dim = 0)

        # preprocess input before passing to residual layers
        output = self.maxpool(self.relu(self.bn1(self.conv1(input))))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output


# for debugging
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
#     model = PoseEncoder( ); print(model)

#     H = 192; W = 640
#     summary(model = model, 
#             input_size = (1, 3, H, W),
#             col_names=["input_size", "output_size", "trainable"],
#             # col_names=["input_size", "output_size", "num_params", "trainable"],
#             col_width=20,
#             row_settings=["var_names"]
#     )
