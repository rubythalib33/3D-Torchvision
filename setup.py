from setuptools import setup, find_packages

setup(
    name='torchvision_3d',
    version='0.2.0',
    description='3D CNN for PyTorch with imagenet pretrained models',
    author='Ruby Abdullah',
    author_email='rubyabdullah14@gmail.com',
    url='https://https://github.com/rubythalib33/3D-Torchvision',
    install_requires=[
        'torch',
        'torchvision',],
    packages=['torchvision_3d'],
    python_requires=">=3.6",
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],
)