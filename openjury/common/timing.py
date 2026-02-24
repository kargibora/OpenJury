from __future__ import annotations

import time


class Timeblock:
    """Timer context manager."""

    def __init__(self, name: str | None = None, verbose: bool = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        if self.verbose:
            print(self)

    def __str__(self):
        name = self.name if self.name else "block"
        return f"{name} took {self.duration} seconds"
