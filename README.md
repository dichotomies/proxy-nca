
# About

This repository contains a PyTorch implementation of [`No Fuss Distance Metric Learning using Proxies`](https://arxiv.org/pdf/1703.07464.pdf) as introduced by Google Research.

The training and evaluation setup is exactly the same as described in the paper, except that Adam was used as optimizer instead of RMSprop.

I have ported the [PyTorch BN-Inception model from PyTorch 0.2](https://github.com/Cadene/pretrained-models.pytorch) to PyTorch >= 0.4. It's weights are stored inside the repository in the directory `net`.

You need Python3, PyTorch >= 1.1 and torchvision >= 0.3.0 to run the code. I have used CUDA Version 10.0.130.

Note that negative log with softmax is used as ProxyNCA loss. Therefore, the anchor-positive-proxy distance is not excluded in the denominator. In practice, I have not noticed a difference.

The importance of scaling of the normalized proxies and embeddings is mentioned in the ProxyNCA paper (in the theoretical background), but the exact scaling factors are ommitted. I have found that (3, 3) work well for CUB and Cars and (8, 1) work well for SOP (first being for proxies and latter for embeddings).

# Reproducing Results

You can adjust most training settings (learning rate, optimizer, criterion, dataset, ...) in the config file. 

You'll only have to adjust the root paths for the datasets. Then you're ready to go.

## Downloading and Extracting the Datasets

### [Cars 196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

```
mkdir cars196
cd cars196
wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
tar -xzvf car_ims.tgz
pwd # use this path as root path for config file
```

### [CUB 200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzvf CUB_200_2011.tgz
cd CUB_200_2011
pwd # use this path as root path for config file
```

### [SOP](https://cvgl.stanford.edu/projects/lifted_struct/)

```
wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
unzip
cd Stanford_Online_Products
pwd # use this path as root path for config file
```

## Commands

```
DATA=cub; SCALING_X=3.0; SCALING_P=3.0; LR=1; python3 train.py --data $DATA \
--log-filename $DATA-scaling_x_$SCALING_X-scaling_p_$SCALING_P-lr_$LR \
--config config.json --epochs=20 --gpu-id 0 --lr-proxynca=$LR \
--scaling-x=$SCALING_X --scaling-p=$SCALING_P --with-nmi
```

```
DATA=cars; SCALING_X=3.0; SCALING_P=3.0; LR=1; python3 train.py --data $DATA \
--log-filename $DATA-scaling_x_$SCALING_X-scaling_p_$SCALING_P-lr_$LR \
--config config.json --epochs=50 --gpu-id 1 --lr-proxynca=$LR \
--scaling-x=$SCALING_X --scaling-p=$SCALING_P --with-nmi
```

```
DATA=sop; SCALING_X=1; SCALING_P=12; LR=10; python3 train.py --data $DATA \
--log-filename $DATA-scaling_x_$SCALING_X-scaling_p_$SCALING_P-lr_$LR \
--config config.json --epochs=50 --gpu-id 3 --lr-proxynca=$LR \
--scaling-x=$SCALING_X --scaling-p=$SCALING_P
```

## Results

The results were obtained mostly with one Titan X or a weaker GPU.

Reading: This Implementation [[Google's Implementation](https://arxiv.org/pdf/1703.07464.pdf)].

|          | CUB               | Cars              | SOP                 |
| -------- | ----------------- | ----------------- | ------------------- |
| Duration | 00:19h            | 00:24h            | 01:55h              |
| Epoch    | 17                | 15                | 16                  |
| Log      | [here](https://github.com/dichotomies/proxy-nca/blob/master/log/cub-scaling_x_3.0-scaling_p_3.0-lr_1.log)              | [here](https://github.com/dichotomies/proxy-nca/blob/master/log/cars-scaling_x_3.0-scaling_p_3.0-lr_1.log)              | [here](https://github.com/dichotomies/proxy-nca/blob/master/log/sop-scaling_x_1-scaling_p_8-lr_10.log)                |
| R@1      | **52.63** [49.21] | 72.19 [**73.22**] | **74.07** [73.73]   |
| R@2      | **64.63** [61.90] | 81.31 [**82.42**] | 79.13 [-------]     |
| R@4      | **75.76** [67.90] | **87.54** [86.36] | 83.30 [-------]     |
| R@8      | **84.52** [72.40] | **92.54** [88.68] | 86.66 [-------]     |
| NMI      | **60.64** [59.53] | 62.45 [**64.90**] | ----------          |

# Referencing this Implementation

If you'd like to reference this ProxyNCA implementation, you can use this bibtex:
 
```
@misc{Tschernezki2020,
  author = {Tschernezki, Vadim and Sanakoyeu, Artsiom},
  title = {PyTorch Implementation of ProxyNCA},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dichotomies/proxy-nca}},
}
```
