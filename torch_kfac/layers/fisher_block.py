from typing import Iterable, Optional, Tuple
import torch
from torch import Tensor

from ..utils import MovingAverageVariable, Lock, compute_pi_adjusted_damping, \
    inverse_by_cholesky, kronecker_product, normalize_damping


class FisherBlock(object):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, device: torch.device, **kwargs):
        self._in_features = in_features
        self._out_features = out_features
        self._dtype = dtype
        self._device = device

        self._activations_cov = MovingAverageVariable((in_features, in_features), dtype=dtype, device=device)
        self._sensitivities_cov = MovingAverageVariable((out_features, out_features), dtype=dtype, device=device)
        self._activations_cov_inv = None
        self._sensitivities_cov_inv = None

        self._forward_lock = False
        self._backward_lock = False

    def setup(self, forward_lock: Lock, backward_lock: Lock, **kwargs) -> None:
        self._forward_lock = forward_lock
        self._backward_lock = backward_lock

    def update_cov(self, cov_ema_decay: float = 1.0) -> None:
        raise NotImplementedError()

    def update_cov_inv(self, damping: Tensor):
        """
        Updates the inverse covariance matrices.
        Args:
            damping: The damping (Tikhonov regularization) to apply before calculating the inverse.
        """
        if self.activation_covariance is None or self.sensitivity_covariance is None:
            raise ValueError("Activations or Sensitivity covariance matrix are None")

        act_cov, sen_cov = self.activation_covariance, self.sensitivity_covariance
        a_damp, s_damp = self.compute_damping(damping, self.renorm_coeff)

        self._activations_cov_inv = inverse_by_cholesky(act_cov, a_damp)
        self._sensitivities_cov_inv = inverse_by_cholesky(sen_cov, s_damp)

    def compute_damping(self, damping: torch.Tensor, normalization: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if normalization is not None:
            maybe_normalized_damping = normalize_damping(damping, normalization)
        else:
            maybe_normalized_damping = damping

        return compute_pi_adjusted_damping(
            self.activation_covariance,
            self.sensitivity_covariance,
            maybe_normalized_damping ** 0.5
        )

    def full_fisher_block(self):
        left_factor = self.activation_covariance
        right_factor = self.sensitivity_covariance
        return self._renorm_coeff * kronecker_product(left_factor, right_factor)
    
    def reset(self) -> None:
        self._activations_cov.reset()
        self._sensitivities_cov.reset()

    def grads_to_mat(self, grads: Iterable[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def mat_to_grads(self, mat_grads: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def multiply(self, grads: Iterable[torch.Tensor], damping: torch.Tensor) -> Iterable[torch.Tensor]:
        act_cov, sen_cov = self.activation_covariance, self.sensitivity_covariance
        a_damp, s_damp = self.compute_damping(damping, self.renorm_coeff)
        act_cov += torch.eye(act_cov.shape[0], device=a_damp.device) * a_damp
        sen_cov += torch.eye(sen_cov.shape[0], device=a_damp.device) * s_damp

        mat_grads = self.grads_to_mat(grads)
        nat_grads = sen_cov @ mat_grads @ act_cov / self.renorm_coeff

        return self.mat_to_grads(nat_grads)
        
    def multiply_preconditioner(self, grads: Iterable[torch.Tensor], damping: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        Multiplies the gradients with the preconditioning matrix, which corresponds to
        calculating S^{1} * g * A^{-1}, where S is the sensitivities covariance matrix
        and A is the activations covariance matrix.
        Args:
            grads: An iterable of the gradients for this block.
            damping: The damping (Tikhonov regularization) to apply before calculating the inverse.
            update_inverses: If true the inverse of the covariance matrices are recalculated.
                If False, the cached inverse covariance matrices are used.
                If the inverse covariance matrices are None, then they are always calculated.
        Returns:
            The preconditioned gradients.
        """
        if self._activations_cov_inv is None or self._sensitivities_cov_inv is None:
            self.update_cov_inv(damping)
        
        mat_grads = self.grads_to_mat(grads)
        nat_grads = self._sensitivities_cov_inv @ mat_grads @ self._activations_cov_inv / self.renorm_coeff

        return self.mat_to_grads(nat_grads)

    @property
    def is_static(self) -> bool:
        return (self._in_features == 0) or (self._out_features == 0)
        
    @property
    def activation_covariance(self) -> torch.Tensor:
        return self._activations_cov.value

    @property
    def sensitivity_covariance(self) -> torch.Tensor:
        return self._sensitivities_cov.value

    @property
    def vars(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError()

    @property
    def grads(self) -> Iterable[torch.Tensor]:
        return [tensor.grad for tensor in self.vars]

    def set_gradients(self, new_grads):
        for var, grad in zip(self.vars, new_grads):
            var.grad.data = grad
    
    @property
    def renorm_coeff(self) -> float:
        return 1.


class ExtensionFisherBlock(FisherBlock):
    def __init__(self, module: torch.nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self.module = module
        self.forward_hook_handle = None
        self.backward_hook_handle = None

        self.forward_hook_handle = self.module.register_forward_hook(self._forward_hook_wrapper)
        self.backward_hook_handle = self.module.register_backward_hook(self._backward_hook_wrapper)

    def _forward_hook_wrapper(self, *args):
        if self._forward_lock:
            return self.forward_hook(*args)

    def _backward_hook_wrapper(self, *args):
        if self._backward_lock:
            return self.backward_hook(*args)

    def forward_hook(self, module: torch.nn.Module, *args):
        raise NotImplementedError()

    def backward_hook(self, module: torch.nn.Module, *args):
        raise NotImplementedError()
