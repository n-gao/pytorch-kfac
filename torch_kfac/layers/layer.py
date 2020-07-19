from torch_kfac.utils.utils import compute_pi_adjusted_damping, inverse_by_cholesky, kronecker_product, normalize_damping
from typing import Iterable, Tuple
import torch

from ..utils import MovingAverageVariable


class Layer(object):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, **kwargs):
        self._in_features = in_features
        self._out_features = out_features
        self._dtype = dtype

        self._activations_cov = MovingAverageVariable((in_features, in_features), dtype=dtype)
        self._sensitivities_cov = MovingAverageVariable((out_features, out_features), dtype=dtype)

    def setup(self, **kwargs) -> None:
        return

    def update_cov(self) -> None:
        raise NotImplementedError()

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
        act_cov += torch.eye(act_cov.shape[0]) * a_damp
        sen_cov += torch.eye(sen_cov.shape[0]) * s_damp

        mat_grads = self.grads_to_mat(grads)
        nat_grads = sen_cov @ mat_grads @ act_cov / self.renorm_coeff

        return self.mat_to_grads(nat_grads)
        
    def multiply_preconditioner(self, grads: Iterable[torch.Tensor], damping: torch.Tensor) -> Iterable[torch.Tensor]:
        act_cov, sen_cov = self.activation_covariance, self.sensitivity_covariance
        a_damp, s_damp = self.compute_damping(damping, self._activations.shape[1])
        act_cov_inverse = inverse_by_cholesky(act_cov, a_damp)
        sen_cov_inverse = inverse_by_cholesky(sen_cov, s_damp)
        
        mat_grads = self.grads_to_mat(grads)
        nat_grads = sen_cov_inverse @ mat_grads @ act_cov_inverse / self.renorm_coeff

        return self.mat_to_grads(nat_grads)
        
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
