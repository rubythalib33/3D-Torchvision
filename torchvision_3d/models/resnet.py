import torch
import torch.nn as nn
import torch.nn.functional as F

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
        assert type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 'type only support for resnet18, resnet34, resnet50, resnet101, resnet152'
        self.type = type
        self.features = None

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
                        pass
                
                features.append(nn.Sequential(*features_child))
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == '__main__':
    from torchvision.models.resnet import resnet18
    sample = torch.randn(1,3,1,224,224)
    model = ResNet3D('resnet50')
    model_18 = resnet18(pretrained=False)
    print(model)
    print(model(sample).shape)