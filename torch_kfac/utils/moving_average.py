import torch


class MovingAverageVariable(object):
    def __init__(self, shape, dtype=None, normalize_value=True):
        if dtype is None:
            dtype = torch.get_default_dtype()
        self._normalize_value = normalize_value
        self._var = torch.zeros(shape, dtype=dtype, requires_grad=False)
        self._total_weight = torch.zeros((), dtype=dtype, requires_grad=False)

    @property
    def dtype(self):
        return self._var.dtype

    @property
    def value(self):
        if self._normalize_value:
            return self._var / self._total_weight
        else:
            return self._var.clone()

    def add_to_average(self, value, decay=1.0, weight=1.0):
        self._var *= decay
        self._total_weight *= decay
        self._var += value
        self._total_weight += weight

    def reset(self):
        self._var = torch.zeros_like(self._var)
        self._total_weight = torch.zeros_like(self._total_weight)
    