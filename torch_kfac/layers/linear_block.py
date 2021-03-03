from typing import Iterable
import torch
from torch.nn import Linear

from .fisher_block import ExtensionFisherBlock, FisherBlock
from ..utils import center, compute_cov, append_homog


class FullyConnectedFisherBlock(ExtensionFisherBlock):
    def __init__(self, module: Linear, **kwargs) -> None:
        super().__init__(
            module=module,
            in_features=module.in_features + int(module.bias is not None),
            out_features=module.out_features,
            dtype=module.weight.dtype,
            device=module.weight.device,
            **kwargs)

        self._activations = None
        self._sensitivities = None
        self._center = False

    @torch.no_grad()
    def forward_hook(self, module: Linear, inp: torch.Tensor, out: torch.Tensor) -> None:
        x = inp[0].detach().clone().reshape(-1, self._in_features - self.has_bias).requires_grad_(False)
        if self._activations is None:
            self._activations = x
        else:
            self._activations = torch.cat([self._activations, x])

    @torch.no_grad()
    def backward_hook(self, module: Linear, grad_inp: torch.Tensor, grad_out: torch.Tensor) -> None:
        x = grad_out[0].clone().detach().reshape(-1, self._out_features).requires_grad_(False) * grad_out[0].shape[0]
        if self._sensitivities is None:
            self._sensitivities = x
        else:
            self._sensitivities = torch.cat([self._sensitivities, x])

    def setup(self, center: bool = False, **kwargs):
        super().setup(**kwargs)
        self._center = center

    def update_cov(self, cov_ema_decay: float = 1.0) -> None:
        if self._activations is None or self._sensitivities is None:
            return
        act, sen = self._activations, self._sensitivities
        if self._center:
            act = center(act)
            sen = center(sen)
        
        if self.has_bias:
            act = append_homog(act)

        activation_cov = compute_cov(act)
        sensitivity_cov = compute_cov(sen)
        self._activations_cov.add_to_average(activation_cov, cov_ema_decay)
        self._sensitivities_cov.add_to_average(sensitivity_cov, cov_ema_decay)

    def grads_to_mat(self, grads: Iterable[torch.Tensor]) -> torch.Tensor:
        if self.has_bias:
            weights, bias = grads
            mat_grads = torch.cat([weights, bias[:, None]], -1)
        else:
            mat_grads = grads[0]
        return mat_grads

    def mat_to_grads(self, mat_grads: torch.Tensor) -> torch.Tensor:
        if self.has_bias:
            return mat_grads[:, :-1], mat_grads[:, -1]
        else:
            return mat_grads,

    @property
    def has_bias(self) -> None:
        return self.module.bias is not None

    @property
    def vars(self) -> Iterable[torch.Tensor]:
        if self.has_bias:
            return [self.module.weight, self.module.bias]
        else:
            return [self.module.weight]
