from typing import Iterable, Union
import numpy as np
import torch
from torch.nn import Conv1d, Conv2d, Conv3d
import torch.nn.functional as F

from .fisher_block import ExtensionFisherBlock
from ..utils import append_homog, center, compute_cov


class ConvFisherBlock(ExtensionFisherBlock):
    def __init__(
        self, 
        module: Union[Conv1d, Conv2d, Conv3d],
        **kwargs
    ) -> None:
        in_features = np.prod(module.kernel_size) * module.in_channels + int(module.bias is not None)
        out_features = module.out_channels
        super().__init__(
            module=module,
            in_features=in_features,
            out_features=out_features, 
            dtype=module.weight.dtype, 
            device=module.weight.device,
            **kwargs
        )
        self.n_dim = len(module.kernel_size)

        self._activations = None
        self._sensitivities = None

        self._center = False

    @torch.no_grad()
    def forward_hook(self, module: Union[Conv1d, Conv2d, Conv3d], inp: torch.Tensor, out: torch.Tensor) -> None:
        self._activations = self.extract_patches(inp[0])

    @torch.no_grad()
    def backward_hook(self, module: Union[Conv1d, Conv2d, Conv3d], grad_inp: torch.Tensor, grad_out: torch.Tensor) -> None:
        self._sensitivities = grad_out[0].transpose(1, -1).contiguous()
        # Reshape to (batch_size, n_spatial_locations, n_out_features)
        self._sensitivities = self._sensitivities.view(
            self._sensitivities.shape[0],
            -1,
            self._sensitivities.shape[-1]
        ) #* self._sensitivities.shape[0]
        # in the original code they scale by the batch_size, I don't quite understand this
        # sometimes it boosts the performance sometimes it hurts

    def setup(self, center: bool = False, **kwargs) -> None:
        self._center = center
        super().setup(**kwargs)

    def update_cov(self) -> None:
        if self._activations is None or self._sensitivities is None:
            return
        act, sen = self._activations, self._sensitivities
        act = act.reshape(-1, act.shape[-1])
        sen = sen.reshape(-1, sen.shape[-1])
        if self._center:
            act = center(act)
            sen = center(sen)

        if self.has_bias:
            act = append_homog(act)

        activation_cov = compute_cov(act)
        sensitivity_cov = compute_cov(sen)
        self._activations_cov.add_to_average(activation_cov)
        self._sensitivities_cov.add_to_average(sensitivity_cov)

    def grads_to_mat(self, grads: Iterable[torch.Tensor]) -> torch.Tensor:
        if self.has_bias:
            weights, bias = grads
            # reshape to (out_features, in_features)
            weights = weights.view(weights.shape[0], -1)
            mat_grads = torch.cat([weights, bias[:, None]], -1)
        else:
            # reshape to (out_features, in_features)
            mat_grads = grads[0].view(grads.shape[0], -1)
        return mat_grads

    def mat_to_grads(self, mat_grads: torch.Tensor) -> torch.Tensor:
        if self.has_bias:
            return mat_grads[:, :-1].view_as(self.module.weight), mat_grads[:, -1]
        else:
            return mat_grads.view_as(self.module.weight),

    @property
    def renorm_coeff(self) -> float:
        return self._activations.shape[1]

    @property
    def has_bias(self) -> bool:
        return self.module.bias is not None

    @property
    def vars(self) -> Iterable[torch.Tensor]:
        if self.has_bias:
            return (self.module.weight, self.module.bias)
        else:
            return (self.module.weight,)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        # Extract convolutional patches
        # Input: (batch_size, in_channels, spatial_dim1, ...)
        # Add padding
        if sum(self.module.padding) > 0:
            padding_mode = self.module.padding_mode
            if padding_mode == 'zeros':
                padding_mode = 'constant'
            x = F.pad(x, tuple(pad for pad in self.module.padding[::-1] for _ in range(2)), mode=padding_mode, value=0.)
        # Unfold the convolution
        for i, (size, stride) in enumerate(zip(self.module.kernel_size, self.module.stride)):
            x = x.unfold(i+2, size, stride)
        # Move in_channels to the end
        # https://github.com/pytorch/pytorch/issues/36048
        x = x.unsqueeze(2+self.n_dim).transpose(1, 2+self.n_dim).squeeze(1)
        # Make the memory contiguous
        x = x.contiguous()
        # Return the shape (batch_size, n_spatial_locations, n_in_features)
        x = x.view(
            x.shape[0],
            np.prod([x.shape[1+i] for i in range(self.n_dim)]),
            -1
        )
        return x
