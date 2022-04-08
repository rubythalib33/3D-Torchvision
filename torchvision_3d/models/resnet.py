import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, base_width=64, groups = 1, downsample=None, model_instance=None):
        super(Bottleneck3D, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv3d(inplanes, width, kernel_size=[1,1,1], bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=[1,3,3], stride=[1,*stride],
                               padding=[0,1,1], bias=False, groups=groups)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=[1,1,1], bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if model_instance is not None:
            self.conv1.weight.data = torch.stack([model_instance.conv1.weight.data], dim=2)
            self.conv2.weight.data = torch.stack([model_instance.conv2.weight.data], dim=2)
            self.conv3.weight.data = torch.stack([model_instance.conv3.weight.data], dim=2)
            
            if self.downsample is not None:
                self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * 4, kernel_size=[1,1,1], stride=[1,*model_instance.downsample[0].stride], padding=[0,*model_instance.downsample[0].padding], bias=False),
                nn.BatchNorm3d(planes * 4))
                self.downsample[0].weight.data = torch.stack([model_instance.downsample[0].weight.data], dim=2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, model_instance=None):
        super(BasicBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=[1,3,3], stride=[1,*stride], padding=[0,1,1], bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=[1,3,3], stride=[1,1,1], padding=[0,1,1], bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

        if model_instance is not None:
            self.conv1.weight.data = torch.stack([model_instance.conv1.weight.data], dim=2)
            self.conv2.weight.data = torch.stack([model_instance.conv2.weight.data], dim=2)
            if self.downsample is not None:
                self.downsample = nn.Sequential(
                    nn.Conv3d(model_instance.downsample[0].in_channels, model_instance.downsample[0].out_channels, kernel_size=[1,1,1], stride=[1,*model_instance.downsample[0].stride], padding=[0,*model_instance.downsample[0].padding], bias=False),
                    nn.BatchNorm3d(model_instance.downsample[0].out_channels)
                )
                self.downsample[0].weight.data = torch.stack([model_instance.downsample[0].weight.data], dim=2)
            
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, type, pretrained=False):
        super(ResNet3D, self).__init__()
        assert type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'], 'type only support for resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2'
        self.type = type
        self.features = None
        
        self.groups = 1
        self.base_width = 64

        if type == 'resnet18':
            from torchvision.models.resnet import resnet18
            self.features = self.init_layer(resnet18(pretrained=pretrained))
        elif type == 'resnet34':
            from torchvision.models.resnet import resnet34
            self.features = self.init_layer(resnet34(pretrained=pretrained))
        elif type == 'resnet50':
            from torchvision.models.resnet import resnet50
            self.features = self.init_layer(resnet50(pretrained=pretrained))
        elif type == 'resnet101':
            from torchvision.models.resnet import resnet101
            self.features = self.init_layer(resnet101(pretrained=pretrained))
        elif type == 'resnet152':
            from torchvision.models.resnet import resnet152
            self.features = self.init_layer(resnet152(pretrained=pretrained))
        elif type == 'resnext50_32x4d':
            from torchvision.models.resnet import resnext50_32x4d
            self.features = self.init_layer(resnext50_32x4d(pretrained=pretrained))
            self.groups = 32
            self.base_width = 4
        elif type == 'resnext101_32x8d':
            from torchvision.models.resnet import resnext101_32x8d
            self.features = self.init_layer(resnext101_32x8d(pretrained=pretrained))
            self.groups = 32
            self.base_width = 8
        elif type == 'wide_resnet50_2':
            from torchvision.models.resnet import wide_resnet50_2
            self.features = self.init_layer(wide_resnet50_2(pretrained=pretrained))
            self.base_width = 64 * 2
        elif type == 'wide_resnet101_2':
            from torchvision.models.resnet import wide_resnet101_2
            self.features = self.init_layer(wide_resnet101_2(pretrained=pretrained))
            self.base_width = 64 * 2

        self.features = self.init_features()
    
    def init_layer(self, model):
        layers = []
        for m in model.children():
            if not isinstance(m, nn.AdaptiveAvgPool2d) and not isinstance(m, nn.Linear):
                layers.append(m)
        return nn.Sequential(*layers)
    
    def init_features(self):
        features = []
        for model in self.features.children():
            if isinstance(model, nn.Conv2d):
                model_temp = nn.Conv3d(in_channels=model.in_channels, out_channels=model.out_channels, kernel_size=(1,*model.kernel_size), stride=(1,*model.stride), padding=(0,*model.padding), bias=False)
                model_temp.weight.data = torch.stack([model.weight.data] , dim=2)
                features.append(model_temp)
            elif isinstance(model, nn.MaxPool2d):
                model_temp = nn.MaxPool3d(kernel_size=[1,model.kernel_size, model.kernel_size], stride=[1,model.stride, model.stride], padding=[0,model.padding, model.padding])
                features.append(model_temp)
            elif isinstance(model, nn.ReLU):
                features.append(model)
            elif isinstance(model, nn.BatchNorm2d):
                model_temp = nn.BatchNorm3d(num_features=model.num_features)
                features.append(model_temp)
            elif isinstance(model, nn.Sequential):
                features_child = []
                for child in model.children():
                    if child._get_name() == 'BasicBlock':
                        child_temp = BasicBlock3D(inplanes=child.conv1.in_channels, planes=child.conv1.out_channels, stride=child.conv1.stride, downsample=child.downsample, model_instance=child)
                        features_child.append(child_temp)
                    elif child._get_name() == 'Bottleneck':
                        child_temp = Bottleneck3D(inplanes=child.conv1.in_channels, planes=child.conv3.out_channels//4, stride=child.conv2.stride, downsample=child.downsample, groups=self.groups, base_width=self.base_width, model_instance=child)
                        features_child.append(child_temp)
                
                features.append(nn.Sequential(*features_child))
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == '__main__':
    from torchvision.models.resnet import resnet50
    sample = torch.randn(1,3,1,224,224)
    model = ResNet3D('resnet101')
    print(model(sample).shape)