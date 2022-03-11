import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet3D(nn.Module):
    def __init__(self, type, pretrained=False):
        super().__init__()
        
        if type == 'densenet121':
            from torchvision.models import densenet121
            model_instance = densenet121(pretrained=pretrained)
            self.features = model_instance.features
        elif type == 'densenet161':
            from torchvision.models import densenet161
            model_instance = densenet161(pretrained=pretrained)
            self.features = model_instance.features
        elif type == 'densenet169':
            from torchvision.models import densenet169
            model_instance = densenet169(pretrained=pretrained)
            self.features = model_instance.features
        elif type == 'densenet201':
            from torchvision.models import densenet201
            model_instance = densenet201(pretrained=pretrained)
            self.features = model_instance.features
        else:
            raise NotImplementedError('type only support for densenet121, densenet161, densenet169, densenet201')
        
        self.features = self.init_features()

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
            elif model._get_name() == '_DenseBlock':
                print('auuch')
            elif model._get_name() == '_Transition':
                print('auuchT')

        return nn.Sequential(*features)


if __name__ == '__main__':
    from torchvision.models import densenet121
    model = densenet121(pretrained=False)
    print(model)
    print('='*50)
    model = DenseNet3D('densenet121', pretrained=False)
    print(model)