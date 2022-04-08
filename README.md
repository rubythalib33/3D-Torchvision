# 3D-Torchvision
3D torchvision with ImageNet Pretrained

![](./assets/logo.png)

## it can be used for:
1. Video Embedding
2. Action Recognition
3. Video task neural network backbone

## How to Setup
```bash
https://github.com/rubythalib33/3D-Torchvision
cd 3D-Torchvision
python setup.py install
```

## Update v.0.2.0
1. add more type for resnet3d model: ```resnext50_32x4d, resnext101_32x8d, wide_resnet50_2 wide_resnet101_2'```
2. support for densenet3d, squeezenet3d and mobilenetv2_3d

## Model Ready
```python
# 1. AlexNet3D:
from torchvision_3d.models import AlexNet3D:

model = AlexNet3D(pretrained=True)

# 2. VGG3D
from torchvision_3d.models import VGG3D

model = VGG3D(type='vgg11', pretrained=True) #type can be vgg11, vgg16, vgg19, vgg11_bn, vgg16_bn, vgg19_bn

#3. ResNet3D
from torchvision_3d.models import ResNet3D

model = ResNet3D(type='resnet50', pretrained=True) #type can be resnet18, resnet34, resnet50, resnet101, resnet152, , resnext50_32x4d, resnext101_32x8d, wide_resnet50_2 wide_resnet101_2'

#4. DenseNet3D
from torchvision_3d.models import DenseNet3D
model = DenseNet3D(type='densenet121', pretrained=True) #type can be densenet121, densenet161, densenet169, densenet201

#5. MobileNetV2_3D
from torchvision_3d.models import MobileNetV2_3D
model = MobileNetV2_3D(pretrained=True)

#6 SqueezeNet3D
from torchvision_3d.models import SqueezeNet3D
model = SqueezeNet3D(type= 'squeezenet1_0', pretrained=True) #type can be squeezenet1_0, squeezenet1_1
```