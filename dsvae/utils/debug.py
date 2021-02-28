import os


class Debug:
    def __init__(self, on: bool = False):
        self._on = on
        # self.is_ipython()
        self.is_pytest()
        self.is_env()

    def __bool__(self) -> bool:
        return self._on

    def is_ipython(self) -> None:
        if get_ipython().__class__.__name__ == "NoneType":
            self._on = True

    def is_env(self) -> None:
        if bool(int(os.environ["DEBUG"])):
            self._on = True

    def is_pytest(self) -> None:
        try:
            if bool(int(os.environ["_PYTEST_RAISE"])):
                self._on = True
        except KeyError:
            pass
