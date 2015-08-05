# kaggle-plankton-torch-7-vgg

1. Get a torch/cuda/GPU-equipped machine or just use this preconfigured AMI: ami-c79b7eac (notes here: https://github.com/brotchie/torch-ubuntu-gpu-ec2-install)
2. Grab the kaggle data here: https://www.kaggle.com/c/datasciencebowl/data
3. Do preprocessing
```
th -i provider.lua
```
```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```
4. Train the model
```
CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg
```