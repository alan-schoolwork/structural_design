import inspect
from typing import Callable

import jax
from jax.experimental.checkify import Error, checkify


def handle_err(err: Error):
    msg = err.get()
    if msg is not None:
        print("checkify: assertion failed")
        print(msg)
        print()


def checkify_simple[**P, R](f: Callable[P, R]) -> Callable[P, R]:

    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        err, out = checkify(f)(*args, **kwargs)
        jax.debug.callback(handle_err, err)
        return out

    setattr(inner, "__signature__", inspect.signature(f))
    return inner
