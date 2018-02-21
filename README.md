
# Proxy NCA

This repository contains a PyTorch implementation of [`No Fuss Distance Metric Learning using Proxies`](https://arxiv.org/pdf/1703.07464.pdf) as introduced by Google Research with minimal boiler-plate.

The Proxy NCA model can be initialized with `pnca.py`. This file also contains an execution with example data, s.t. it is clear what kind of data the model expects. Furthermore, Inception V3 was used as no-top-model.

The notebook demonstrates an example training of the model with CUB 200. I think this makes it quite straight forward to adapt this implementation to any other dataset.
