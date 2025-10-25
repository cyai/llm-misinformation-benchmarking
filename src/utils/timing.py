import time
from contextlib import contextmanager


@contextmanager
def timing_ms():
    start = time.perf_counter()
    try:
        yield lambda: int((time.perf_counter() - start) * 1000)
    finally:
        ...
