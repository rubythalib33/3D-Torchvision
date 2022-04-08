from turtle import forward
import torch
import torch.nn as nn

class Fire3D(nn.Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        model = None
    ) -> None:
        super(Fire3D, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze.weight.data = torch.stack([model.squeeze.weight.data] , dim=2) if model is not None else self.squeeze.weight.data
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1.weight.data = torch.stack([model.expand1x1.weight.data] , dim=2) if model is not None else self.expand1x1.weight.data
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes, kernel_size=(1,3,3), padding=(0,1,1))
        self.expand3x3.weight.data = torch.stack([model.expand3x3.weight.data] , dim=2) if model is not None else self.expand3x3.weight.data
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet3D(nn.Module):
    def __init__(self, type, pretrained=False):
        super().__init__()
        
        if type not in ['squeezenet1_0', 'squeezenet1_1']:
            raise NotImplementedError('type only support for squeezenet1_0, squeezenet1_1')
        
        if type == 'squeezenet1_0':
            from torchvision.models import squeezenet1_0
            model_instance = squeezenet1_0(pretrained=pretrained)
            self.features = model_instance.features
        elif type == 'squeezenet1_1':
            from torchvision.models import squeezenet1_1
            model_instance = squeezenet1_1(pretrained=pretrained)
            self.features = model_instance.features
            
        self.pretrained = pretrained
        self.features = self.init_features()
    
    def init_features(self):
        features = []
        for model in self.features.children():
            if isinstance(model, nn.Conv2d):
                model_temp = nn.Conv3d(in_channels=model.in_channels, out_channels=model.out_channels, kernel_size=(1,*model.kernel_size), stride=(1,*model.stride), padding=(0,*model.padding), bias=False)
                model_temp.weight.data = torch.stack([model.weight.data] , dim=2) if self.pretrained else model_temp.weight.data
                features.append(model_temp)
            elif isinstance(model, nn.MaxPool2d):
                model_temp = nn.MaxPool3d(kernel_size=[1,model.kernel_size, model.kernel_size], stride=[1,model.stride, model.stride], padding=[0,model.padding, model.padding])
                features.append(model_temp)
            elif isinstance(model, nn.ReLU):
                features.append(model)
            elif isinstance(model, nn.BatchNorm2d):
                model_temp = nn.BatchNorm3d(num_features=model.num_features)
                features.append(model_temp)
            elif model._get_name() == 'Fire':
                inplanes = model.inplanes
                squeeze_planes = model.squeeze.out_channels
                expand1x1_planes = model.expand1x1.out_channels
                expand3x3_planes = model.expand3x3.out_channels
                if self.pretrained:
                    features.append(Fire3D(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, model))
                else:
                    features.append(Fire3D(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes))
            
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.features(x)
        
        return x

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 1, 224, 224)
    model = SqueezeNet3D('squeezenet1_0', pretrained=True)
    outputs = model(inputs)
    print(outputs.shape)
