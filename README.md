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

ResNet-101 trained with our **unsupervised** CMC objective (only contrasting Luminance with Chrominance) surpasses **supervisedly** trained AlexNet on ImageNet classification (60.1% v.s. 59.3%). For this first time on ImageNet classification, unsupervised methods are surpassing the classic supervised-AlexNet proposed in 2012 (CPC++ and DIM++ also achieve this milestone concurrently).

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

**Note:** It seems to us that training with Pytorch version >= 1.0 yields slightly worse results. If you find the similar discrepancy and figure out the problem, please report this since we are trying to fix it as well.

## Training Encoder with CMC on ImageNet

NCE flags:
- `--nce_k`: number of negatives to contrast for each positive. Default: 4096
- `--nce_m`: the momentum for dynamically updating the memory. Default: 0.5
- `--nce_t`: temperature that modulates the distribution. Default: 0.07 for ImageNet, 0.1 for STL-10

Path flags:
- `--data_folder`: specify the ImageNet data folder. 
- `--model_path`: specify the path to save model. 
- `--tb_path`: specify where to save tensorboard monitoring events.

An example of command line for training CMC (Default: AlexNet on Single GPU)
```
CUDA_VISIBLE_DEVICES=0 python train_CMC.py --data_folder path/to/data --model_path path/to/save --tb_path path/to/tensorboard
```

By default, the training scripts will use L and ab as two views to contrast with each other. If you want to specify other image channels as different views, simply modifying [here](https://github.com/HobbitLong/CMC/blob/master/train_CMC.py#L97), 

## Training Linear Classifier

Path flags:
- `--data_folder`: specify the ImageNet data folder. Should be the same as above.
- `--save_path`: specify the path to save the linear classifier. 
- `--tb_path`: specify where to save tensorboard events monitoring linear classifier training.

Specify the checkpoint that you want to evaluate with `--model_path` flag, this path should directly point to the `.pth` file.

Therefore, an example of command line for evaluating, say `./models/ckpt.pth`, should look like:
```
CUDA_VISIBLE_DEVICES=0 python LinearProbing.py --data_folder path/to/data --save_path path/to/save --tb_path path/to/tensorboard --model_path ./models/ckpt.pth
```
<!--
## Results
**ImageNet**: we tabulate the top-1 accuracy (%) of linear probing for different networks trained with CMC on imagenet classification. We also include the supervised AlexNet accuracy for comparison.
|          |Unpervised AlexNet | Unpervised ResNet-50 | Unpervised ResNet-101  | Supervised AlexNet |
|----------|:----:|:---:|:---:|:---:|
| Top-1 | 42.6 | 58.1 | 60.1  | 59.3|
-->

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@inproceedings{tian2019cmc,
  title={Contrastive Multiview Coding},
  author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
  journal={arXiv preprint arXiv:1906.05849},
  year={2019}
}
```
For any questions, please contact Yonglong Tian (yonglong@mit.edu).

## Acknowledgements

Part of this code is inspired by Zhirong Wu's unsupervised learning algorithm [lemniscate](https://github.com/zhirongw/lemniscate.pytorch).
