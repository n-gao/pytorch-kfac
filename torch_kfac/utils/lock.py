from contextlib import contextmanager


class Lock(object):
    def __init__(self) -> None:
        self._entered: bool = False

    @contextmanager
    def __call__(self) -> None:
        assert not self._entered
        try:
            self._entered = True
            yield
        finally:
            self._entered = False

    def __bool__(self) -> bool:
        return self._entered
