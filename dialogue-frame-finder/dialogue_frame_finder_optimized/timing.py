"""
Timing decorator — measures wall-clock duration of any pipeline stage.
"""

import time
import functools


def timed(func):
    """Wrap *func*, injecting ``_stage_duration_sec`` into dict results."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = round(time.perf_counter() - start, 3)
        if isinstance(result, dict):
            result["_stage_duration_sec"] = elapsed
        return result

    return wrapper
