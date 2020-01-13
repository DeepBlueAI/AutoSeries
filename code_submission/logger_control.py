import logging
import sys
import time
import functools
from typing import Any
from contextlib import contextmanager


def get_logger(verbosity_level, name, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(name)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


@contextmanager
def time_limit(pname, verbose=True):
    """limit time"""
    start_time = time.time()
    try:
        if verbose:
            LOGGER.info(f'start {pname}')
        yield
    finally:
        exec_time = time.time() - start_time
    if verbose:
        LOGGER.info(f'{pname} success, time spent {exec_time} sec')


def timeclass(cls):
    def timeit(method, start_log=None):
        @functools.wraps(method)
        def timed(*args, **kw):
            global is_start
            global nesting_level

            is_start = True
            log(f"Start [{cls}.{method.__name__}]:" + (start_log if start_log else ""))
            log(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            nesting_level += 1

            start_time = time.time()
            result = method(*args, **kw)
            end_time = time.time()

            nesting_level -= 1
            log(f"End   [{cls}.{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
            log(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            is_start = False

            return result

        return timed

    return timeit


def log(entry: Any):
    global nesting_level
    nesting_level = 2
    space = "-" * (4 * nesting_level)
    # print(f"{space}{entry}")