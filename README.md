# 3D-Torchvision
3D torchvision with ImageNet Pretrained

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

## Model Ready
```python
# 1. AlexNet3D:
from torchvision_3d.models import AlexNet3D:

model = AlexNet3D

# 2. VGG3D
from torchvision_3d.models import VGG3D

model = VGG3D(type='vgg11') #type can be vgg11, vgg16, vgg19, vgg11_bn, vgg16_bn, vgg19_bn

#3. ResNet3D
from torchvision_3d.models import ResNet3D

model = ResNet3D(type='resnet50') #type can be resnet18, resnet34, resnet50, resnet101, resnet152
```