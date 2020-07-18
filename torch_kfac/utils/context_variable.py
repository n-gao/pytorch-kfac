class ContextVariable(object):
    def __init__(self) -> None:
        self._value = False

    def __enter__(self):
        self._value = True

    def __exit__(self, exc_type, exc_value, traceback):
        self._value = False

    def __call__(self):
        return self.value

    @property
    def value(self):
        return self._value
