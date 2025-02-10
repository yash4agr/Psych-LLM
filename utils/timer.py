import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    """Simple timer context manager"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description} took {elapsed:.2f} seconds")