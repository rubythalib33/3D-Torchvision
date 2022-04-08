import torch
import torch.nn as nn

class ConvNormActivation3D(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding = None,
        groups: int = 1,
        norm_layer = torch.nn.BatchNorm3d,
        activation_layer= torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
        model = None
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv3d(in_channels, out_channels, (1,kernel_size, kernel_size), (1,stride,stride), (0, padding,padding),
                                  dilation=dilation, groups=groups, bias=norm_layer is None)]
        if model is not None:
            layers[0].weight.data = torch.stack([model[0].weight.data] , dim=2)
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels

class InvertedResidual3D(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer = None,
        model = None
    ) -> None:
        super(InvertedResidual3D, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            if model is None:
                layers.append(ConvNormActivation3D(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
            else:
                layers.append(ConvNormActivation3D(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                        activation_layer=nn.ReLU6, model=model[0]))
        if model is None:
            layers.extend([
                # dw
                ConvNormActivation3D(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                                activation_layer=nn.ReLU6),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ])
        else:
            pw_linear = nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False)
            pw_linear.weight.data = torch.stack([model.conv[2].weight.data] , dim=2)
            layers.extend([
                # dw
                ConvNormActivation3D(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                                activation_layer=nn.ReLU6, model=model[1]),
                # pw-linear
                pw_linear,
                norm_layer(oup),
            ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_3D(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v2

        self.pretrained = pretrained

        model_instance = mobilenet_v2(pretrained=pretrained)
        self.features = model_instance.features
        self.features = self.init_features()
    
    def init_features(self):
        features = []
        for model in self.features.children():
            if model._get_name() == 'ConvNormActivation':
                in_channels = model[0].in_channels
                out_channels = model[0].out_channels
                kernel_size = model[0].kernel_size[0]
                stride = model[0].stride[0]
                padding = model[0].padding[0] if model[0].padding is not None else None
                dilation = model[0].dilation[0]
                if self.pretrained:
                    layer = ConvNormActivation3D(in_channels=in_channels, out_channels=out_channels, dilation=dilation, activation_layer=nn.ReLU6, kernel_size=kernel_size, stride=stride, padding=padding, model=model)
                else:
                    layer = ConvNormActivation3D(in_channels=in_channels, out_channels=out_channels, dilation=dilation, activation_layer=nn.ReLU6, kernel_size=kernel_size, stride=stride, padding=padding)
                features.append(layer)
            elif model._get_name() == 'InvertedResidual':
                inp = model.conv[0][0].in_channels
                oup = model.conv[-2].out_channels
                stride = model.stride
                expand_ratio = model.conv[-2].in_channels / inp
                if self.pretrained:
                    layer = InvertedResidual3D(inp, oup, stride, expand_ratio)
                else:
                    layer = InvertedResidual3D(inp, oup, stride, expand_ratio)
                features.append(layer)
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        for model in self.features.children():
            x = model(x)
        return x

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 1, 224, 224)
    model = MobileNetV2_3D(pretrained=True)
    outputs = model(inputs)
    print(outputs.shape)