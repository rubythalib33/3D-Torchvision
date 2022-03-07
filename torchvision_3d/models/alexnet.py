from torchvision.models import alexnet
from torch import conv3d, nn
import torch

class AlexNet3D(nn.Module):
    def __init__(self, pretrained=False):
        super(AlexNet3D, self).__init__()
        self.alexnet_model = alexnet(pretrained=pretrained)
        self.alexnet_model = self.alexnet_model.features
        self.features = self.init_features()

    def forward(self, x):
        return self.features(x)

    def init_features(self):
        features = []
        for model in self.alexnet_model.children():
            if isinstance(model, nn.Conv2d):
                model_temp = nn.Conv3d(in_channels=model.in_channels, out_channels=model.out_channels, kernel_size=(1,*model.kernel_size), stride=(1,*model.stride), padding=(0,*model.padding))
                model_temp.weight.data = torch.stack([model.weight.data] , dim=2)
                model_temp.bias.data = model.bias.data
                features.append(model_temp)
            elif isinstance(model, nn.MaxPool2d):
                model_temp = nn.MaxPool3d(kernel_size=[1,model.kernel_size, model.kernel_size], stride=[1,model.stride, model.stride], padding=[0,model.padding, model.padding])
                features.append(model_temp)
            elif isinstance(model, nn.ReLU):
                features.append(model)
            elif isinstance(model, nn.BatchNorm2d):
                model_temp = nn.BatchNorm3d(num_features=model.num_features)
                features.append(model_temp)
        
        return nn.Sequential(*features)
if __name__ == '__main__':
    example = torch.randn(1, 3, 3, 224, 224)
    model = AlexNet3D()
    print(model(example).shape)