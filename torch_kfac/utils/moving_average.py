import torch


class MovingAverageVariable(object):
    def __init__(
            self,
            shape: torch.Size,
            dtype: torch.dtype = None,
            device: torch.device = None,
            normalize_value: bool = True) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        self._normalize_value = normalize_value
        self._var = torch.zeros(
            shape, dtype=dtype, device=device, requires_grad=False)
        self._total_weight = torch.zeros(
            (), dtype=dtype, device=device, requires_grad=False)

    @property
    def dtype(self) -> torch.dtype:
        return self._var.dtype

    @property
    def value(self) -> torch.Tensor:
        if self._normalize_value:
            return self._var / self._total_weight
        else:
            return self._var.clone()

    def add_to_average(self, value: torch.Tensor, decay: float = 1.0, weight: float = 1.0) -> None:
        self._var *= decay
        self._total_weight *= decay
        self._var += value
        self._total_weight += weight

    def reset(self) -> None:
        self._var = torch.zeros_like(self._var)
        self._total_weight = torch.zeros_like(self._total_weight)

    @value.setter
    def value(self, new_value) -> None:
        if self._normalize_value:
            self._var.data = new_value * self._normalize_value
        else:
            self._var.data = new_value
