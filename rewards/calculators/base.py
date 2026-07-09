import os

class Calculator():
    def __init__(
        self,
        task: str,
    ) -> None:
        self.task = task

    def calc(self):
        NotImplementedError
