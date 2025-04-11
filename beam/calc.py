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
from lib.columns import calc_buckle, calc_inertia
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

# np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(precision=10, suppress=False)


class Vars_moe(eqx.Module):
    b: fval
    h: fval

    W: fval
    L: fval

    deflection: fval = eqx.field(default_factory=lambda: (1 / 8 * areg.inch))

    __repr__ = pformat_repr


def get_moe(v: Vars_moe):

    I = v.b * v.h**3 / 12

    ans = v.W * v.L**3 / 48 / I / v.deflection

    return convert_unit(ans, areg.psi)


def experiments_moe() -> Iterable[tuple[str, Vars_moe]]:
    yield "(1/8 by 1/4) default", Vars_moe(
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
        W=260 * areg.force_grams,
        L=7.0 * areg.inch,
    )

    yield "(1/8 by 1/4) min", Vars_moe(
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
        W=260 * areg.force_grams,
        L=5.5 * areg.inch,
    )

    yield "(1/8 by 1/4) max", Vars_moe(
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
        W=260 * areg.force_grams,
        L=8 * areg.inch,
    )

    yield "(1/2 by 1/4) default", Vars_moe(
        b=1 / 2 * areg.inch,
        h=1 / 4 * areg.inch,
        W=400 * areg.force_grams,
        L=14.0 * areg.inch,
    )

    yield "(1/2 by 1/4) noticeably weak", Vars_moe(
        b=1 / 2 * areg.inch,
        h=1 / 4 * areg.inch,
        W=400 * areg.force_grams,
        L=9.5 * areg.inch,
    )

    yield "(1/2 by 1/4) planned to maybe go on beam", Vars_moe(
        b=1 / 2 * areg.inch,
        h=1 / 4 * areg.inch,
        W=660 * areg.force_grams,
        L=11.0 * areg.inch,
        deflection=1 / 16 * areg.inch,
    )


class Vars_stress(eqx.Module):
    # distance between ends
    d: fval
    # object weight
    w: fval

    # cross sections of balsa
    b: fval
    h: fval

    __repr__ = pformat_repr


def get_stress(v: Vars_stress):

    S = v.b * v.h**2 / 6

    fail_mom = v.d / 2 * v.w / 2

    return convert_unit(fail_mom / S, areg.psi)


def experiments_stress() -> Iterable[tuple[str, Vars_stress]]:

    yield "a", Vars_stress(
        d=23.0 * areg.inch,
        w=400 * areg.force_grams,
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
    )

    yield "b", Vars_stress(
        d=18.0 * areg.inch,
        w=400 * areg.force_grams,
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
    )

    yield "a2", Vars_stress(
        d=16.0 * areg.inch,
        w=160.0 * areg.force_grams,
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
    )
    yield "b2", Vars_stress(
        d=23.0 * areg.inch,
        w=160.0 * areg.force_grams,
        b=1 / 4 * areg.inch,
        h=1 / 8 * areg.inch,
    )


@allow_autoreload
@unitify
def main():

    # for x, y in experiments_moe():
    #     print(x)
    #     print(y)
    #     print(get_moe(y))
    #     print()
    #     print()

    # for x, y in experiments_stress():
    #     print(x)
    #     print(y)
    #     print(get_stress(y))
    #     print()
    #     print()

    shape = """
    x.x
    x.x
    """

    for moe in [600_000 * areg.psi, 1_000_000 * areg.psi]:
        buckle_strength = calc_buckle(
            shape,
            height=10.5 * areg.inch,
            x_len=1 / 4 * areg.inch,
            modulus_of_elasticity=moe,
            k_factor=1.0,
        )
        print("buckle_strength", convert_unit(buckle_strength, areg.force_pounds))
        print("expect hold", buckle_strength * jnp.sin(21.8 * areg.degrees) * 2)

    for stress in [6000 * areg.psi, 8000 * areg.psi]:
        print(
            "crush_strength",
            convert_unit(
                1 / 2 * areg.inch * 1 / 2 * areg.inch * stress, areg.force_pounds
            ),
        )

    for stress in [6000 * areg.psi, 8000 * areg.psi]:
        v = convert_unit(
            1 / 4 * areg.inch * 1 / 8 * areg.inch * 4 * stress, areg.force_pounds
        )
        print("tension strengh", v)
        print("expect hold", v * jnp.tan(21.8 * areg.degrees) * 2)
