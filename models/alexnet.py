from torchvision.models import alexnet
from torch import conv3d, nn
import torch

alexnet_model = alexnet(pretrained=True)
alexnet_model = alexnet_model.features

class AlexNet3D(nn.Module):
    def __init__(self):
        super(AlexNet3D, self).__init__()
        self.features = self.init_features()

    def forward(self, x):
        return self.features(x)

    def init_features(self):
        features = []
        for model in alexnet_model.children():
            if isinstance(model, nn.Conv2d):
                model_temp = nn.Conv3d(in_channels=model.in_channels, out_channels=model.out_channels, kernel_size=model.kernel_size[0], stride=model.stride[0], padding=model.padding[0])
                model_temp.weight.data = torch.stack([model.weight.data] * model.kernel_size[0], dim=2)
                model_temp.bias.data = model.bias.data
                features.append(model_temp)
            elif isinstance(model, nn.MaxPool2d):
                model_temp = nn.MaxPool3d(kernel_size=model.kernel_size, stride=model.stride, padding=model.padding)
                features.append(model_temp)
            elif isinstance(model, nn.ReLU):
                features.append(model)
        
        return nn.Sequential(*features)
if __name__ == '__main__':
    example = torch.randn(1, 3, 224, 224, 224)
    model = AlexNet3D()
    print(model(example).shape)