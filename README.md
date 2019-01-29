
# About

This repository contains a PyTorch implementation of [`No Fuss Distance Metric Learning using Proxies`](https://arxiv.org/pdf/1703.07464.pdf) as introduced by Google Research.

The setup is the same as in the paper, except that Adam was used as optimizer instead of RMSprop. In particular, the sizes of the embeddings and batches equal 64 and 32 respectively. Also, [BN-Inception](http://arxiv.org/abs/1502.03167) is used and trained with random resized crop and horizontal flip and evaluated with resized center crop. 

I have ported the [PyTorch BN-Inception model from PyTorch 0.2](https://github.com/Cadene/pretrained-models.pytorch) to 0.4. It's weights are stored inside the repository in the directory `net`.

You need Python3 and minimum PyTorch 0.4.1 to run the code.

Note that cross entropy with softmax is used for calculating the actual ProxyNCA loss. Therefore, the anchor-positive-proxy distance is not excluded in the denominator. In practice, this makes no difference (probably due to the high amount of classes for the datasets). I've also included the actual ProxyNCA (`ProxyNCAUnstable`) loss (you can choose whether the anchor-positive distance is excluded in the denominator). It works a bit slower.

# Reproducing Results with [CUB 200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [Cars 196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

You can adjust the training settings (learning rate, optimizer, criterion, dataset, ...) in the config files (`config.json` (for CUB), `config_cars.json` (for Cars)). 

You'll only have to adjust the root paths for the CUB and Cars dataset; then you're ready to go.

## Downloading and Extracting the Datasets

### Cars196

```
mkdir cars196
cd cars196
wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
tar -xzvf car_ims.tgz
pwd # use this path as root path in config file
```

### CUB200-2011
```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzvf CUB_200_2011.tgz
cd CUB_200_2011
pwd # use this path as root path in config file
```

## Results - CUB

Training takes about 15 seconds per epoch with one Titan X (Pascal). You should get decent results (R@1 > 51) after 7 epochs (less than 2 minutes).

```
python3 train.py --data cub --log-filename test-cub --config config.json --gpu-id 0
```

| Metric | This Implementation  | [Google's Implementation](https://arxiv.org/pdf/1703.07464.pdf) |
| ------ | -------------------- | ------------- |
|  R@1   |       **52.46**      |     49.21     |
|  R@2   |       **64.78**      |     61.90     |
|  R@4   |       **75.38**      |     67.90     |
|  R@8   |       **84.31**      |     72.40     |
|  NMI   |       **60.84**      |     59.53     |

An example training log file can be found in the log dir, see [`29-01-19-cub.log`](https://raw.githubusercontent.com/dichotomies/proxy-nca/master/log/29-01-19-cub.log).

## Results - Cars

```
python3 train.py --data cars --log-filename test-cars --config config_cars.json --gpu-id 0
```

Training takes about 35 seconds per epoch with one Titan X (Pascal). The model converges at about 70 epochs with the LR used in the config file. You might want to reduce the LR and train for more epochs to get a higher recall.

| Metric | This Implementation  | [Google's Implementation](https://arxiv.org/pdf/1703.07464.pdf) |
| ------ | -------------------- | ------------- |
|  R@1   |         71.12        |   **73.22**   |
|  R@2   |         80.03        |   **82.42**   |
|  R@4   |       **86.74**      |     86.36     |
|  R@8   |       **92.01**      |     88.68     |
|  NMI   |         61.70        |   **64.90**   |

An example training log file can be found in the log dir, see [`29-01-19-cars.log`](https://raw.githubusercontent.com/dichotomies/proxy-nca/master/log/29-01-19-cars.log).
