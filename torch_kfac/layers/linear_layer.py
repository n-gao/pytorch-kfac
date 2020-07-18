from typing import Iterable
from torch_kfac.utils.utils import compute_pi_adjusted_damping
import torch
import torch.nn as nn
from torch.nn import Linear

from .layer import Layer
from ..utils import center, compute_cov, append_homog, inverse_by_cholesky


class LinearLayer(Layer):
    def __init__(self, module: Linear, **kwargs) -> None:
        self.module = module
        self._center = False
        super().__init__(
            in_features=module.in_features + self.has_bias,
            out_features=module.out_features,
            dtype=module.weight.dtype,
            **kwargs)

        self._activations = None
        self._sensitivities = None

        def forward_hook(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
            self._activations = inp[0].clone().detach().reshape(-1, self._in_features - self.has_bias).requires_grad_(False)

        def backward_hook(module: nn.Module, grad_inp: torch.Tensor, grad_out: torch.Tensor) -> None:
            self._sensitivities = grad_out[0].clone().detach().reshape(-1, self._out_features).requires_grad_(False)
        
        self.forward_hook_handle = self.module.register_forward_hook(forward_hook)
        self.backward_hook_handle = self.module.register_backward_hook(backward_hook)

    def setup(self, center: bool = False, **kwargs):
        self._center = center

    def update_cov(self) -> None:
        act, sen = self._activations, self._sensitivities
        if self._center:
            act = center(act)
            sen = center(sen)
        
        if self.has_bias:
            act = append_homog(act)

        activation_cov = compute_cov(act)
        sensitivity_cov = compute_cov(sen)
        self._activations_cov.add_to_average(activation_cov)
        self._sensitivities_cov.add_to_average(sensitivity_cov)

    def multiply_preconditioner(self, grads: Iterable[torch.Tensor], damping: torch.Tensor) -> Iterable[torch.Tensor]:
        act_cov, sen_cov = self.activation_covariance, self.sensitivity_covariance
        a_damp, s_damp = self.compute_damping(damping)
        act_cov_inverse = inverse_by_cholesky(act_cov, a_damp)
        sen_cov_inverse = inverse_by_cholesky(sen_cov, s_damp)

        # Concatenate weights and biases
        if self.has_bias:
            weights, bias = grads
            grads = torch.cat([weights, bias[:, None]], -1)
        else:
            grads = grads[0]

        nat_grads = sen_cov_inverse @ grads @ act_cov_inverse

        # Split up again
        if self.has_bias:
            return nat_grads[:, :-1], nat_grads[:, -1]
        else:
            return nat_grads,

    @property
    def has_bias(self) -> None:
        return self.module.bias is not None

    @property
    def vars(self) -> Iterable[torch.Tensor]:
        if self.has_bias:
            return [self.module.weight, self.module.bias]
        else:
            return [self.module.weight]
