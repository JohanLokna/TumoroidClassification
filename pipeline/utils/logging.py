from warnings import warn
from typing import Callable


class BaseLogger:

    def __init__(self, logger: Callable):
        self.logger = logger

    def __call__(self, message: str):
        self.logger(message)


class StandardLogger(BaseLogger):

    def __init__(self):
        super().__init__(lambda m: warn(m))


class PrintLogger(BaseLogger):

    def __init__(self):
        super().__init__(lambda m: print(m))
