
# About

This repository contains a PyTorch implementation of [`No Fuss Distance Metric Learning using Proxies`](https://arxiv.org/pdf/1703.07464.pdf) as introduced by Google Research.

The same parameters were used as described in the paper, except for the optimizer. In particular, the size of the embedding and batches equals 64 and 32 respectively. Also, [Inception V2](http://arxiv.org/abs/1502.03167) is used and trained with random resized crop and horizontal flip and evaluated with resized center crop. 

I have ported the [PyTorch Inception V2 model from PyTorch 0.2](https://github.com/Cadene/pretrained-models.pytorch) to 0.4. It's weights are stored inside the repository in the directory `net`.

# Reproducing Results with CUB 200

The only thing to reproduce the results in the table below is to execute: `python3 train.py`.

In this case, the CUB dataset will be automatically downloaded to the directory `cub200` (default) and verified with the corresponding md5 hash. If you train the model for the first time, then the images file will be extracted automatically in the same folder. After that you can use the argument `--cub-is-extracted` to avoid extracting the dataset over and over again.

| Metric | This Implementation  | [Google's Implementation](https://arxiv.org/pdf/1703.07464.pdf) |
| ------ | -------------------- | ------------- |
|  R@1   |       **49.26**      |     49.21     |
|  R@2   |         60.99        |   **61.90**   |
|  R@4   |       **71.31**      |     67.90     |
|  R@8   |       **80.78**      |     72.40     |
|  NMI   |         58.12        |   **59.53**   |

An example training log file can be found in the log dir, see [`example.log`](https://github.com/dichotomies/proxy-nca/raw/master/log/example.log).
