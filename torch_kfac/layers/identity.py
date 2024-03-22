from typing import Iterable
import torch

from .fisher_block import FisherBlock


class Identity(FisherBlock):
    def __init__(self, module: torch.nn.Module, **kwargs) -> None:
        self.module = module
        super().__init__(
            in_features=0,
            out_features=0,
            dtype=torch.get_default_dtype(),
            device='cpu',
            **kwargs)

    def update_cov(self, cov_ema_decay: float = 1.0):
        return

    def multiply(self, grads: Iterable[torch.Tensor], damping: torch.Tensor) -> Iterable[torch.Tensor]:
        return grads

    def multiply_preconditioner(self, grads: Iterable[torch.Tensor], damping: torch.Tensor,
                                update_inverses: bool) -> Iterable[torch.Tensor]:
        return grads
        
    @property
    def normalization_factor(self):
        return 1.

    @property
    def vars(self) -> Iterable[torch.Tensor]:
        return tuple()
