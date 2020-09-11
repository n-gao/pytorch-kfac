# PyTorch KFAC
This repository is meant to port most of the original KFAC [repository](https://github.com/tensorflow/kfac) to PyTorch.

## Why should I use KFAC?
KFAC is an approximation of the natural gradients, i.e., it is an approximate second-order method allowing for larger step sizes at the cost of compute time. This should enable faster convergence.

## Usage

### Installation
```shell
git clone https://github.com/n-gao/pytorch-kfac.git
cd pytorch-kfac
python setup.py install
```

### Example
An example notebook is provided in `examples`.

One has to enable the tracking of the forward pass and backward pass.
This prevents the optimizer to track other passes, e.g., for validation.
```python
kfac = torch_kfac.KFAC(model, learning_rate=1e-3, damping=1e-3)
...
model.zero_grad()
with kfac.track_forward():
  # forward pass
  loss = ...
with kfac.track_backward():
  # backward pass
  loss.backward()
kfac.step()

# Do some validation
```

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

## TODOs
* Due to the different structure of the KFAC optimizer, implementing the `torch.optim.optimizer.Optimizer` interface could only work by hacks, but you are welcome to add this.
This implies that `torch.optim.lr_scheduler` are not available for KFAC. However, the learning rate can still be manually be edited by changing the `.learning_rate` property.

* Currently it the optimizer assumes that the output is reduced by `mean`, other reductions are not supported yet. So if you use e.g. `torch.nn.CrossEntropyLoss` make sure to have
`reduction='mean'`.

* Add more examples

* Add support for shared layers (,e.g., RNNs)

* Documentation

* Add support for distributed training (This is partially possible by synchronizing the covariance matrices and gradients manually.)

* `optimizer.zero_gard` is not available yet, you can use `model.zero_grad`

## References
### Orignal Repsotiry
[tensorflow/kfac](https://github.com/tensorflow/kfac)

### Literature
This work is based on:
* [Martens et Grosse's 'Optimizing Neural Networks with Kronecker-factored Approximate Curvature'](https://arxiv.org/abs/1503.05671)
* [Grosse et Marten's 'A Kronecker-factored approximate Fisher matrix for convolution layers'](https://arxiv.org/abs/1602.01407)
