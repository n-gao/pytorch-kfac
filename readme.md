# PyTorch KFAC
This repository is meant to port most of the original KFAC [repository](https://github.com/tensorflow/kfac) to PyTorch.

## Why should I use KFAC?
KFAC is an approximation of the natural gradients, i.e., it is an approximate second-order method allowing for larger step sizes at the cost of compute time. This should enable faster convergence.

## Usage
An example notebook is provided in `examples`.

## Features
This implementation features the following features:
* Regular and Adam momentum
* Adaptive damping
* Weight decay
* Norm constraint

The following layers are currently supported:
* `nn.Linear`
* `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`

Further, the code is designed to be easily extendible by implementing the `torch_kfac.layers.Layer` interface with a custom layer.
Unsupported layers fall back to classical gradient descent.

## Drawbacks
Due to the different structure of the KFAC optimizer, implementing the `torch.optim.optimizer.Optimizer` interface could only work by hacks, but you are welcome to add this.
This implies that `torch.optim.lr_scheduler` are not available for KFAC. However, the learning rate can still be manually be edited by changing the `.learning_rate` property.

Currently it the optimizer assumes that the output is reduced by `mean`, other reductions are not supported yet. So if you use e.g. `torch.nn.CrossEntropyLoss` make sure to have
`reduction='mean'`.

## References
### Orignal Repsotiry
[tensorflow/kfac](https://github.com/tensorflow/kfac)

### Literature
This work is based on:
* [Martens et Grosse's 'Optimizing Neural Networks with Kronecker-factored Approximate Curvature'](https://arxiv.org/abs/1503.05671)
* [Grosse et Marten's 'A Kronecker-factored approximate Fisher matrix for convolution layers'](https://arxiv.org/abs/1602.01407)
