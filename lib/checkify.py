import inspect
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import oryx
from beartype import beartype as typechecker
from jax import tree_util as jtu
from jax._src.typing import ArrayLike, DType, Shape
from jax.experimental.checkify import Error, check, checkify
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped
from pintax import areg, get_value, unitify, ureg
from pintax.functions import lstsq
from pintax.unstable import convert_unit

from lib.beam import force_profile, force_profile_builder
from lib.jax_utils import flatten_handler
from lib.lstsq import flstsq, flstsq_checked
from lib.plot import plot_unit
from lib.utils import bval, fval, ival, jit


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
