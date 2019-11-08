Note: this repo currently has not covered the most recent `ResNet50x2` model in the `Table2` of our arXiv paper. I will add it in a couple of weeks, once I succeed a bunch of recent deadlines.


## Contrastive Multiview Coding

This repo covers the implementation for CMC, which learns representations from multiview data in a self-supervised way (by multiview, we mean multiple sensory, multiple modal data, or literally multiple viewpoint data. It's flexible to define what is a "view"):

"Contrastive Multiview Coding" [Paper](http://arxiv.org/abs/1906.05849), [Project Page](http://hobbitlong.github.io/CMC/).

![Teaser Image](http://hobbitlong.github.io/CMC/CMC_files/teaser.jpg)

## Highlights

**(1) Representation quality as a function of number of contrasted views.** 

We found that, the more views we train with, the better the representation (of each single view).

**(2) Contrastive objective v.s. Predictive objective**

We compare the contrastive objective to cross-view prediction, finding an advantage to the contrastive approach.

**(3) Unsupervised v.s. Supervised**

ResNet-50 trained with our **unsupervised** CMC objective surpasses **supervisedly** trained AlexNet on ImageNet classification ( ~63.0% v.s. 59.3%). For this first time on ImageNet classification, unsupervised methods are surpassing the classic supervised-AlexNet proposed in 2012 (CPC++ and DIM++ also achieve this milestone concurrently). 

## Updates

Aug 20, 2019 - ResNets on ImageNet have been added.

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

**Note:** It seems to us that training with Pytorch version >= 1.0 yields slightly worse results. If you find the similar discrepancy and figure out the problem, please report this since we are trying to fix it as well.

## Training AlexNet/ResNets with CMC on ImageNet

**Note:** For AlexNet, we split across the channel dimension and use each half to encode L and ab. For ResNets, we use a standard ResNet model to encode each view.

NCE flags:
- `--nce_k`: number of negatives to contrast for each positive. Default: 4096
- `--nce_m`: the momentum for dynamically updating the memory. Default: 0.5
- `--nce_t`: temperature that modulates the distribution. Default: 0.07 for ImageNet, 0.1 for STL-10

Path flags:
- `--data_folder`: specify the ImageNet data folder.
- `--model_path`: specify the path to save model.
- `--tb_path`: specify where to save tensorboard monitoring events.

Model flag:
- `--model`: specify which model to use, including *alexnet*, *resnet50*, and *resnet101*

An example of command line for training CMC (Default: AlexNet on Single GPU)
```
CUDA_VISIBLE_DEVICES=0 python train_CMC.py --data_folder path/to/data \
 --model_path path/to/save \
 --tb_path path/to/tensorboard
```

Training CMC with ResNets requires at least 4 GPUs with DataParallel, the command of using *resnet50* looks like
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_CMC.py --data_folder path/to/data \
 --model_path path/to/save \
 --tb_path path/to/tensorboard \
 --model resnet50 --batch_size 128 --crop_low 0.08
```

By default, the training scripts will use L and ab as two views to contrast with each other. If you want to specify other image channels as different views, simply modifying the image transform function [here](https://github.com/HobbitLong/CMC/blob/master/train_CMC.py#L101) and changing the mean and std accordingly should work.

## Training Linear Classifier

Path flags:
- `--data_folder`: specify the ImageNet data folder. Should be the same as above.
- `--save_path`: specify the path to save the linear classifier.
- `--tb_path`: specify where to save tensorboard events monitoring linear classifier training.

Model flag `--model` is similar as above and should be specified.

Specify the checkpoint that you want to evaluate with `--model_path` flag, this path should directly point to the `.pth` file.

This repo provides 3 ways to train the linear classifier: *single GPU*, *data parallel*, and *distributed data parallel*.

(a) *single GPU.*
An example of command line for evaluating, say `./models/alexnet.pth`, should look like:
```
CUDA_VISIBLE_DEVICES=0 python LinearProbing.py --data_folder path/to/data \
 --save_path path/to/save \
 --tb_path path/to/tensorboard \
 --model_path ./models/alexnet.pth \
 --gpu 0 \
 --model alexnet --learning_rate 0.1 --layer 5
```
(b) *data parallel.*
The command is similar as (a) except for specifying multiple gpus and removing the `--gpu 0` flag.

(c) *distributed data parallel.*
This way is typically faster than ordinary data parallel even when using the same number of GPUs on a single node. Therefore, it's recommended if you have extra GPUs. An example of the command line looks like:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python LinearProbing.py --data_folder path/to/data \
 --save_path path/to/save \
 --tb_path path/to/tensorboard \
 --model_path ./models/alexnet.pth \
 --dist-url 'tcp://127.0.0.1:7788' --dist-backend 'nccl' \
 --multiprocessing-distributed --world-size 1 --rank 0 \
 --model alexnet --learning_rate 0.1 --layer 5
```

**Note:** When training linear classifiers on top of ResNets, it's important to use large learning rate, e.g., 30~50. Specifically, change `--model alexnet --learning_rate 0.1 --layer 5` to `--model resnet50 --learning_rate 30 --layer 6`.

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{tian2019contrastive,
  title={Contrastive Multiview Coding},
  author={Tian, Yonglong and Krishnan, Dilip and Isola, Phillip},
  journal={arXiv preprint arXiv:1906.05849},
  year={2019}
}
```
For any questions, please contact Yonglong Tian (yonglong@mit.edu).

## Acknowledgements

Part of this code is inspired by Zhirong Wu's unsupervised learning algorithm [lemniscate](https://github.com/zhirongw/lemniscate.pytorch).
