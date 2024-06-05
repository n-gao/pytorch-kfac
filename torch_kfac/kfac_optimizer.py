import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple
import torch

from .layers import init_fisher_block, FisherBlock
from .utils import Lock, inner_product_pairs, scalar_product_pairs


class KFAC(object):
    def __init__(self,
                 model: torch.nn.Module,
                 learning_rate: float,

                 damping: torch.Tensor,
                 adapt_damping: bool = False,
                 damping_adaptation_decay: float = 0.99,
                 damping_adaptation_interval: int = 5,
                 include_damping_in_qmodel_change: bool = False,
                 min_damping=1e-8,

                 cov_ema_decay: float = 0.95,

                 momentum: float = 0.9,
                 momentum_type: str = 'regular',

                 norm_constraint: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 l2_reg: float = 0.,

                 update_cov_manually: bool = False,
                 center: bool = False,
                 enable_pi_correction: bool = True) -> None:
        """Creates the KFAC Optimizer object.

        Args:
            model (torch.nn.Module): A `torch.nn.Module` to optimize.
            learning_rate (float): The initial learning rate
            damping (torch.Tensor): This quantity times the identiy matrix is (approximately) added
                to the matrix being estimated. - This relates to the "trust" in the second order approximation.
            adapt_damping (bool, optional): If True we adapt the damping according to the Levenberg-Marquardt
                rule described in Section 6.5 of the original K-FAC paper. The details of this scheme are controlled by
                various additional arguments below. Defaults to False.
            damping_adaptation_decay (float, optional): The `damping` parameter is multiplied by the `damping_adaption_decay`
                every `damping_adaption_interval` number of iterations. Defaults to 0.99.
            damping_adaptation_interval (int, optional): Number of steps in between updating the `damping` parameter. Defaults to 5.
            include_damping_in_qmodel_change (bool, optional): If True the damping contribution is included in the quadratic model
                for the purposes of computing qmodel_change in rho. Defaults to False.
            min_damping ([type], optional): Minimum value the damping parameter can take. The default is quite arbitrary. Defaults to 1e-8.
            cov_ema_decay (float): The decay factor used when calculating the covariance estimate moving averages. Defaults to 0.95.
            momentum (float, optional): The momentum decay constant to use.. Defaults to 0.9.
            momentum_type (str, optional): The type of momentum to use. Options: [`regular`, `adam`]. Defaults to 'regular'.
            norm_constraint (Optional[float], optional): If specified, the update is scaled down so that its approximate squared
                Fisher norm v^T F v is at most the specified value. May only be used with `regular` momentum. Defaults to None.
            weight_decay (Optional[float], optional): The coefficient to use for weight decay. If set to `None` there is no
                weight decay. Defaults to None.
            l2_reg (float, optional): L2 normalization. Defaults to 0..
            update_cov_manually (bool, optional): If set to `True`, the covariance matrices are not updated automatically at every
                `.step()` call. You will have to call it manually using `.update_cov()`. This is useful in distributed settings
                or when you want your covariances w.r.t. the model distribution rather than the loss function. Defaults to False.
            center (bool, optional): If set to True the activations and sensitivities are centered. This is useful when dealing with
                unnormalized distributions. Defaults to False.
            enable_pi_correction (bool, optional): If set to true, the pi-correction for the Tikhonov regularization
                will be calculated.
        """

        legal_momentum_types = ['regular', 'adam']
        momentum_type = momentum_type.lower()
        assert momentum_type in legal_momentum_types, f'{momentum_type} type momentum is not supported.'
        assert momentum_type not in [
            'regular' 'adam'] or norm_constraint is None, 'Norm constraint may only be used with regular momentum.'

        self.model = model
        self.blocks: List[FisherBlock] = []
        self.learning_rate = learning_rate

        self.counter = 0

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self._damping = torch.tensor(damping, device=device, dtype=dtype)
        self._adapt_damping = adapt_damping
        self._damping_adaptation_decay = damping_adaptation_decay
        self._damping_adaptation_interval = damping_adaptation_interval
        self._omega = damping_adaptation_decay ** damping_adaptation_interval
        self._include_damping_in_qmodel_change = include_damping_in_qmodel_change
        self._qmodel_change = torch.tensor(np.nan, device=device, dtype=dtype)
        self._prev_loss = torch.tensor(np.nan, device=device, dtype=dtype)
        self._rho = torch.tensor(np.nan, device=device, dtype=dtype)
        self._min_damping = min_damping

        self._weight_decay = weight_decay
        self._l2_reg = l2_reg
        self._norm_constraint = norm_constraint

        self._cov_ema_decay = cov_ema_decay

        self._momentum = momentum
        self._momentum_type = momentum_type

        self.track_forward = Lock()
        self.track_backward = Lock()
        for module in model.modules():
            self.blocks.append(
                init_fisher_block(
                    module,
                    center=center,
                    enable_pi_correction=enable_pi_correction,
                    forward_lock=self.track_forward,
                    backward_lock=self.track_backward
                )
            )

        self._velocities: Dict[FisherBlock, Iterable[torch.Tensor]] = {}
        self.update_cov_manually = update_cov_manually

    def reset_cov(self) -> None:
        for block in self.blocks:
            block.reset_cov()

    def _add_weight_decay(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]) -> Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]:
        """Applies weight decay.
        """
        return tuple(
            (tuple(grad + self._weight_decay*var for grad,
                   var in zip(grads, layer.vars)), layer)
            for grads, layer in grads_and_layers
        )

    def _squared_fisher_norm(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]], precon_grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]) -> float:
        """Computes the squared (approximate) Fisher norm of the updates.

        This is defined as v^T F v, where F is the approximate Fisher matrix
        as computed by the estimator, and v = F^{-1} g, where g is the gradient.
        This is computed efficiently as v^T g.
        """
        return inner_product_pairs(grads_and_layers, precon_grads_and_layers)

    def _update_clip_coeff(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]], precon_grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]) -> float:
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
        sq_norm_grad = self._squared_fisher_norm(
            grads_and_layers, precon_grads_and_layers)
        sq_norm_up = sq_norm_grad * self.learning_rate**2
        return torch.min(
            torch.ones((), dtype=sq_norm_up.dtype, device=sq_norm_up.device),
            torch.sqrt(self._norm_constraint / sq_norm_up)
        )

    def _clip_updates(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]], precon_grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]) -> Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]:
        """Rescales the preconditioned gradients to satisfy the norm constraint.

        Rescales the preconditioned gradients such that the resulting update r
        (after multiplying by the learning rate) will satisfy the norm constraint.
        This constraint is that r^T F r <= C, where F is the approximate Fisher
        matrix, and C is the norm_constraint attribute. See Section 5 of
        Ba et al., Distributed Second-Order Optimization using Kronecker-Factored
        Approximations.
        """
        coeff = self._update_clip_coeff(
            grads_and_layers, precon_grads_and_layers)
        return scalar_product_pairs(coeff, precon_grads_and_layers)

    def _multiply_preconditioner(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]) -> Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]:
        return tuple((layer.multiply_preconditioner(grads, self.damping), layer) for (grads, layer) in grads_and_layers)

    def _update_velocities(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]], decay: float, vec_coeff=1.0) -> Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]:
        def _update_velocity(grads, layer):
            if layer not in self._velocities:
                self._velocities[layer] = tuple(
                    torch.zeros_like(grad) for grad in grads)
            velocities = self._velocities[layer]

            for velocity, grad in zip(velocities, grads):
                new_velocity = decay * velocity + vec_coeff * grad
                velocity.data = new_velocity
            return velocities

        return tuple((_update_velocity(grads, layer), layer) for grads, layer in grads_and_layers)

    def _compute_approx_qmodel_change(self, updates_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]], grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]) -> torch.Tensor:
        quad_term = 0.5 * inner_product_pairs(updates_and_layers, tuple(
            (layer.multiply(grads, self.damping), layer) for grads, layer in grads_and_layers))
        linear_term = inner_product_pairs(updates_and_layers, grads_and_layers)
        if not self._include_damping_in_qmodel_change:
            quad_term -= 0.5 * self._sub_damping_out_qmodel_change_coeff * \
                self.damping * linear_term
        return quad_term + linear_term

    def _get_raw_updates(self) -> Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]:
        # Get grads
        grads_and_layers = tuple((layer.grads, layer) for layer in self.blocks if any(
            grad is not None for grad in layer.grads))

        if self._momentum_type == 'regular':
            # Multiple preconditioner
            raw_updates_and_layers = self._multiply_preconditioner(
                grads_and_layers)

            # Apply "KL clipping"
            if self.use_norm_constraint:
                raw_updates_and_layers = self._clip_updates(
                    grads_and_layers, raw_updates_and_layers)

            # Update velocities
            if self.use_momentum:
                raw_updates_and_layers = self._update_velocities(
                    raw_updates_and_layers, self._momentum)

            # Do adaptive damping
            if self._adapt_damping and self.is_damping_adaption_time:
                updates_and_layers = scalar_product_pairs(
                    -self.learning_rate,
                    raw_updates_and_layers
                )
                self._qmodel_change = self._compute_approx_qmodel_change(
                    updates_and_layers, grads_and_layers)

        elif self._momentum_type == 'adam':
            # For adam like momentum we first compute the velocities and use the velocities also for KL clipping instead
            # of computing the velocities at the very end.
            # Update velocities
            if self.use_momentum:
                velocities_and_layers = self._update_velocities(
                    grads_and_layers, self._momentum)
            else:
                velocities_and_layers = grads_and_layers

            # Multiply preconditioner
            raw_updates_and_layers = self._multiply_preconditioner(
                grads_and_layers)

            # Apply "KL clipping"
            if self.use_norm_constraint:
                raw_updates_and_layers = self._clip_updates(
                    velocities_and_layers, raw_updates_and_layers)

            # Do adaptive damping
            if self._adapt_damping and self.is_damping_adaption_time:
                # See https://github.com/tensorflow/kfac/blob/cf6265590944b5b937ff0ceaf4695a72c95a02b9/kfac/python/ops/optimizer.py#L1009
                self._qmodel_change = 0.5 * self.learning_rate**2 * inner_product_pairs(raw_updates_and_layers, velocities_and_layers)\
                    - self.learning_rate * \
                    inner_product_pairs(
                        raw_updates_and_layers, grads_and_layers)
        else:
            raise NotImplementedError(
                f'Momentum {self._momentum_type} is not supported yet.')
        return raw_updates_and_layers

    def _update_damping(self, loss):
        # Adapts the damping parameter. KFAC Section 6.5
        if not self._adapt_damping or not self.is_damping_adaption_time:
            return
        loss_change = loss - self._prev_loss
        rho = loss_change / self._qmodel_change

        should_decrease = (
            loss_change < 0 and self._qmodel_change > 0) or rho > 0.75
        should_increase = rho < 0.25

        if should_decrease:
            new_damping = self.damping * self._omega
        elif should_increase:
            new_damping = self.damping / self._omega
        else:
            new_damping = self.damping

        new_damping = torch.clamp(
            new_damping, min=self._min_damping + self._l2_reg)

        self._damping = new_damping
        self._rho = rho

    @torch.no_grad()
    def step(self, loss: Optional[torch.Tensor] = None) -> None:
        if self._adapt_damping and loss is None:
            raise ValueError(
                'The loss must be passed if adaptive damping is used.')

        # We update the damping before the optimization step
        # This allows us to avoid multiple passes through the model
        # and can leave the loss computation out of the optimizer.
        # We shouldn't do this at the very first iteration! (Some variables will be nan)
        self._update_damping(loss)

        # Update covariance matrices
        # We allow for manual updates in case we need more control over the optimization
        # routine, e.g., when distributin KFAC
        if not self.update_cov_manually:
            self.update_cov()

        raw_updates_and_layers = self._get_raw_updates()

        # Apply weight decay
        if self.use_weight_decay:
            raw_updates_and_layers = self._add_weight_decay(
                raw_updates_and_layers)

        # Apply the new gradients
        for precon_grad, layer in raw_updates_and_layers:
            layer.set_gradients(precon_grad)

        # Do gradient step - if any parameter gradient was not updated by its natural gradient
        # this will fall back to the normal gradient.
        for param in self.model.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-self.learning_rate)

        # Cache previous loss
        if loss is not None:
            self._prev_loss = loss.clone()

        self.counter += 1

    def update_cov(self) -> None:
        for layer in self.blocks:
            layer.update_cov(cov_ema_decay=self._cov_ema_decay)

    @property
    def covariances(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                block._activations_cov._var,
                block._sensitivities_cov._var
            )
            for block in self.blocks
            if not block.is_static
        ]

    @covariances.setter
    def covariances(self, new_covariances: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for block, (a_cov, s_cov) in zip(filter(lambda a: not a.is_static, self.blocks), new_covariances):
            block._activations_cov.value = a_cov.to(
                block._activations_cov._var, non_blocking=True)
            block._sensitivities_cov.value = s_cov.to(
                block._sensitivities_cov._var, non_blocking=True)

    @property
    def damping(self) -> torch.Tensor:
        return self._damping.clone()

    @property
    def use_weight_decay(self) -> bool:
        return self._weight_decay is not None and self._weight_decay != 0.

    @property
    def use_norm_constraint(self) -> bool:
        return self._norm_constraint is not None

    @property
    def use_momentum(self) -> bool:
        return self._momentum_type in ['regular', 'adam'] and self._momentum != 0

    @property
    def is_damping_adaption_time(self) -> bool:
        # We do *not* want to update at the first iteration as the previous loss is unknown!
        return ((self.counter+1) % self._damping_adaptation_interval) == 0

    @property
    def _sub_damping_out_qmodel_change_coeff(self) -> float:
        return 1.0 - self._l2_reg / self.damping
