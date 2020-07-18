from torch_kfac.layers.layer import Layer
from typing import Iterable, List, Optional, Tuple
import torch
from torch.optim.optimizer import Optimizer

from .layers import init_layer


class KFAC(object):
    def __init__(self, 
        model: torch.nn.Module, 
        learning_rate: float, 
        damping: torch.Tensor,
        norm_constraint: Optional[float] = None,
        weight_decay: Optional[float] = None,
        center: bool = False) -> None:
        self.model = model
        self.layers: List[Layer] = []
        self.learning_rate = learning_rate

        self._damping = torch.tensor(damping)
        self._weight_decay = weight_decay
        self._norm_constraint = norm_constraint

        for module in model.modules():
            self.layers.append(init_layer(module, center=center))

    def reset_cov(self) -> None:
        for layer in self.layers:
            layer.reset_cov()

    def _add_weight_decay(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], Layer]]) -> Iterable[Tuple[Iterable[torch.Tensor], Layer]]:
        """Applies weight decay.
        """
        return tuple(
            (tuple(grad + self._weight_decay*var for grad, var in zip(grads, layer.vars)), layer)
            for grads, layer in grads_and_layers
        )

    def _squared_fisher_norm(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], Layer]], precon_grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], Layer]]) -> float:
        """Computes the squared (approximate) Fisher norm of the updates.

        This is defined as v^T F v, where F is the approximate Fisher matrix
        as computed by the estimator, and v = F^{-1} g, where g is the gradient.
        This is computed efficiently as v^T g.
        """
        return sum([(g*ng).sum() for (g, _), (ng, _) in zip(grads_and_layers, precon_grads_and_layers)])

    def _update_clip_coeff(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], Layer]], precon_grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], Layer]]) -> float:
        """Computes the scale factor for the update to satisfy the norm constraint.

        Defined as min(1, sqrt(c / r^T F r)), where c is the norm constraint,
        F is the approximate Fisher matrix, and r is the update vector, i.e.
        -alpha * v, where alpha is the learning rate, and v is the preconditioned
        gradient.

        This is based on Section 5 of Ba et al., Distributed Second-Order
        Optimization using Kronecker-Factored Approximations. Note that they
        absorb the learning rate alpha (which they denote eta_max) into the formula
        for the coefficient, while in our implementation, the rescaling is done
        before multiplying by alpha. Hence, our formula differs from theirs by a
        factor of alpha.
        """
        sq_norm_grad = self._squared_fisher_norm(grads_and_layers, precon_grads_and_layers)
        sq_norm_up = sq_norm_grad * self.learning_rate**2
        return torch.min(
            torch.ones((), dtype=sq_norm_up.dtype, device=sq_norm_up.device),
            torch.sqrt(self._norm_constraint / sq_norm_up)
        )

    def _clip_updates(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], Layer]], precon_grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], Layer]]) -> Iterable[Tuple[Iterable[torch.Tensor], Layer]]:
        """Rescales the preconditioned gradients to satisfy the norm constraint.

        Rescales the preconditioned gradients such that the resulting update r
        (after multiplying by the learning rate) will satisfy the norm constraint.
        This constraint is that r^T F r <= C, where F is the approximate Fisher
        matrix, and C is the norm_constraint attribute. See Section 5 of
        Ba et al., Distributed Second-Order Optimization using Kronecker-Factored
        Approximations.
        """
        coeff = self._update_clip_coeff(grads_and_layers, precon_grads_and_layers)
        return tuple((coeff*pg, l) for (pg, l) in precon_grads_and_layers)

    def _update_cov(self) -> None:
        for layer in self.layers:
            layer.update_cov()

    @torch.no_grad()
    def step(self) -> None:
        # Update covariance matrices
        self._update_cov()
        # Get grads
        grads_and_layers = tuple((layer.grads, layer) for layer in self.layers)
        # Apply weight decay
        if self.use_weight_decay:
            grads_and_layers = self._add_weight_decay(grads_and_layers)
        # Multiple preconditioner
        precon_grads_and_layers = tuple((layer.multiply_preconditioner(grads, self.damping), layer) for (grads, layer) in grads_and_layers)
        if self.use_norm_constraint:
            precon_grads_and_layers = self._clip_updates(grads_and_layers, precon_grads_and_layers)
        
        # Apply the new gradients
        for precon_grad, layer in precon_grads_and_layers:
            layer.set_gradients(precon_grad)

        # Do gradient step
        for param in self.model.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-self.learning_rate)

    @property
    def damping(self) -> torch.Tensor:
        return self._damping.clone()

    @property
    def use_weight_decay(self) -> bool:
        return self._weight_decay is not None and self._weight_decay == 0.

    @property
    def use_norm_constraint(self) -> bool:
        return self._norm_constraint is not None
