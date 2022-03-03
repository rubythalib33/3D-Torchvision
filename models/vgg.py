from turtle import forward
import torch.nn as nn
import torch

class VGG3D(nn.Module):
    def __init__(self, type, pretrained=False):
        super(VGG3D, self).__init__()
        assert type in ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'], 'type only support for vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn'
        self.type = type
        self.features = None

        if type == 'vgg11':
            from torchvision.models.vgg import vgg11
            self.features = vgg11(pretrained=pretrained).features
        elif type == 'vgg13':
            from torchvision.models.vgg import vgg13
            self.features = vgg13(pretrained=pretrained).features
        elif type == 'vgg16':
            from torchvision.models.vgg import vgg16
            self.features = vgg16(pretrained=pretrained).features
        elif type == 'vgg19':
            from torchvision.models.vgg import vgg19
            self.features = vgg19(pretrained=pretrained).features
        elif type == 'vgg11_bn':
            from torchvision.models.vgg import vgg11_bn
            self.features = vgg11_bn(pretrained=pretrained).features
        elif type == 'vgg13_bn':
            from torchvision.models.vgg import vgg13_bn
            self.features = vgg13_bn(pretrained=pretrained).features
        elif type == 'vgg16_bn':
            from torchvision.models.vgg import vgg16_bn
            self.features = vgg16_bn(pretrained=pretrained).features
        elif type == 'vgg19_bn':
            from torchvision.models.vgg import vgg19_bn
            self.features = vgg19_bn(pretrained=pretrained).features
        
        self.features = self.init_features()
    
    def forward(self, x):
        return self.features(x)
    
    def init_features(self):
        features = []
        for model in self.features.children():
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
            elif isinstance(model, nn.BatchNorm2d):
                model_temp = nn.BatchNorm3d(num_features=model.num_features)
                features.append(model_temp)
        
        return nn.Sequential(*features)

if __name__ == '__main__':
    example = torch.randn(1, 3, 224, 224, 224).cuda()
    type_ = "vgg11"
    model = VGG3D(type=type_).cuda()
    print(model(example).shape)