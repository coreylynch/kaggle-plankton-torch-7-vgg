# kaggle-plankton-torch-7-vgg

#### What is this?
This is some [torch 7](http://torch.ch/) code to train a [VGG-style](http://arxiv.org/pdf/1409.1556.pdf) convolutional neural network to classify plankton species from a greyscale image. Read more about the dataset [here](https://www.kaggle.com/c/datasciencebowl). This is adapted from the CIFAR torch tutorial [here](http://torch.ch/blog/2015/07/30/cifar.html).

#### Get Data and a Machine
Get a torch/cuda/GPU-equipped machine or just use this preconfigured AMI: ami-c79b7eac (notes here: https://github.com/brotchie/torch-ubuntu-gpu-ec2-install)
Grab the kaggle data here: https://www.kaggle.com/c/datasciencebowl/data
#### Preprocess the Data
```
th -i provider.lua
```
```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```
#### Train the model
```
CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg
```

