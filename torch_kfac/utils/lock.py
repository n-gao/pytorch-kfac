class Lock():
    def __init__(self) -> None:
        self._entered = False

    def __enter__(self) -> None:
        if self._entered:
            raise RuntimeError('Can not enter twice.')
        self._entered = True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._entered:
            raise RuntimeError('Can not exit if never entered.')
        self._entered = False

    def __bool__(self) -> bool:
        return self._entered
