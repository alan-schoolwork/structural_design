from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import oryx
from beartype import beartype as typechecker
from jax import tree_util as jtu
from jax._src.typing import ArrayLike, DType, Shape
from jax.experimental.checkify import check, checkify
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped

from lib.beam import force_profile, force_profile_builder
from lib.checkify import checkify_simple
from lib.jax_utils import debug_print, flatten_handler
from lib.lstsq import flstsq, flstsq_checked
from lib.plot import plot_unit
from lib.utils import (
    allow_autoreload,
    bval,
    debug_callback,
    fval,
    ival,
    jit,
    pformat_repr,
)
from pintax import areg, convert_unit, quantity, unitify, ureg
from pintax._utils import pretty_print
from pintax.functions import lstsq

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", True)

np.set_printoptions(precision=3, suppress=True)


class Vars(eqx.Module):
    b: fval
    h: fval

    W: fval
    L: fval

    deflection: fval = eqx.field(default_factory=lambda: (1 / 8 * areg.inch))

    __repr__ = pformat_repr


def get_moe(v: Vars):

    I = v.b * v.h**3 / 12

    ans = v.W * v.L**3 / 48 / I / (1 / 8 * areg.inch)

    return convert_unit(ans, areg.psi)


def declarations() -> Iterable[tuple[str, Vars]]:
    yield "(1/8 by 1/4) default", Vars(
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
        W=260 * areg.force_grams,
        L=7.0 * areg.inch,
    )

    yield "(1/2 by 1/4) default", Vars(
        b=1 / 2 * areg.inch,
        h=1 / 4 * areg.inch,
        W=400 * areg.force_grams,
        L=14.0 * areg.inch,
    )

    yield "(1/2 by 1/4) noticeably weak", Vars(
        b=1 / 2 * areg.inch,
        h=1 / 4 * areg.inch,
        W=400 * areg.force_grams,
        L=9.5 * areg.inch,
    )

    yield "(1/2 by 1/4) planned to maybe go on beam", Vars(
        b=1 / 2 * areg.inch,
        h=1 / 4 * areg.inch,
        W=660 * areg.force_grams,
        L=11.0 * areg.inch,
        deflection=1 / 16 * areg.inch,
    )


@allow_autoreload
@unitify
def testfn():

    for x, y in declarations():
        print(x)
        print(y)
        print(get_moe(y))
        print()
        print()
