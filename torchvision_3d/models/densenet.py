import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

class _DenseLayer3D(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features, affine=True, eps=1e-05, momentum=0.1, track_running_stats=True))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size *
                                           growth_rate, affine=True, eps=1e-05, momentum=0.1, track_running_stats=True))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size *
                                           growth_rate, growth_rate, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
        self.bn_size = bn_size
    
    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output
    
    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)
    
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock3D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer3D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

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
                num_layers = len(model)
                num_input_features = model.denselayer1.conv1.in_channels
                growth_rate = model.denselayer1.conv2.out_channels
                bn_size = model.denselayer1.conv2.in_channels//growth_rate
                drop_rate = model.denselayer1.drop_rate
                memory_efficient = model.denselayer1.memory_efficient
                layer = _DenseBlock3D(num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient)
                features.append(layer)
            elif model._get_name() == '_Transition':
                num_input_features = model.conv.in_channels
                num_output_features = model.conv.out_channels
                layer = _Transition(num_input_features, num_output_features)
                features.append(layer)

        return nn.Sequential(*features)
    
    def forward(self, x):
        for model in self.features.children():
            x = model(x)
        return x


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 1, 224, 224)
    model = DenseNet3D('densenet121', pretrained=False)
    outputs = model(inputs)
    print(outputs.shape)