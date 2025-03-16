from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar

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
from pintax import areg, convert_unit, quantity, unitify, ureg
from pintax._utils import pretty_print
from pintax.functions import lstsq

from lib.beam import force_profile, force_profile_builder
from lib.checkify import checkify_simple
from lib.jax_utils import debug_print, flatten_handler
from lib.lstsq import flstsq, flstsq_checked
from lib.plot import plot_unit
from lib.utils import bval, debug_callback, fval, ival, jit

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", True)

np.set_printoptions(precision=3, suppress=True)

out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)


@jax.jit
def testfn(v):
    return eqx.error_if(v, v, "error!!!")


@unitify
def tmp():
    f = 15.0 * areg.ksi

    flange_w = 17.6 * areg.inch
    flange_h = 3.54 * areg.inch

    flange_one_area = flange_w * flange_h
    d = 41.1 * areg.inch

    mom_one_area = (flange_one_area * f) * (d / 2 - flange_h / 2)
    mom_two_area = mom_one_area * 2

    debug_print("s", mom_two_area / f)

    # web_w = 1.97 * areg.inch

    # mom_web = web_w * d - flange_h


@unitify
def main():
    calc_beam(version_=1)
    calc_beam(version_=2)
    calc_beam(version_=3)
    calc_beam(version_=4)
    calc_beam(version_=5)


@partial(jit, static_argnames=["version_"])
@checkify_simple
@unitify
def calc_beam(version_: ival):

    version = int(version_)
    debug_print(f"version: {version}")

    pf = areg.pound_force

    tag_force = "force"

    @jit
    @checkify_simple
    def build_prof() -> force_profile:
        builder = force_profile_builder(-0.001 * areg.ft, 60.001 * areg.ft, 100000)

        builder.add_uniform(
            0.0 * areg.ft,
            60.0 * areg.ft,
            -(100.0 * pf / areg.ft**2) * (30.0 * areg.ft),
        )

        def add_points(*points):
            for i, p in enumerate(points):
                builder.add_point(
                    p * areg.ft, oryx.core.sow(0.0 * pf, name=f"p{i}", tag=tag_force)
                )

        if version == 1:
            add_points(0.0, 60.0)
        elif version == 2:
            add_points(15.0, 45.0)
        elif version == 3:
            tmp = 30 / (jnp.sqrt(2) + 1)
            add_points(tmp, 60.0 - tmp)
        elif version == 4:
            add_points(30.0)
        elif version == 5:
            # add_points(10.0, 45.0)
            add_points(10.0, 30.0, 50.0)

        return builder.build()

    force_vars: dict[str, Array]
    force_vars = oryx.core.reap(build_prof, tag=tag_force)()
    force_tree, force_vars_l = flatten_handler.create(force_vars)

    def prof_fn(args_l: Array) -> force_profile:
        args = force_tree.unflatten(args_l)
        return oryx.core.plant(build_prof, tag=tag_force)(plants=args)

    def get_eqns(args_l: Array):
        prof = prof_fn(args_l)
        return prof.net_force() / areg.kip, prof.net_rot() / (areg.kip * areg.ft)

    lstsq_res = flstsq_checked(get_eqns, force_vars_l)
    # debug_callback(lambda x: print(pretty_print(x).format()), lstsq_res)
    forces = lstsq_res.x
    debug_print("forces", convert_unit(forces, areg.kip))

    if version == 1:
        check(jnp.allclose(forces[0], forces[1], atol=0.0), "version 1 symmetry")
        width = convert_unit(
            forces[0] / (15.0 * areg.ksi) / (1.0 * areg.inch), areg.inch
        )
        debug_print("steel hanger", width)

    prof = prof_fn(forces)

    mom = convert_unit(prof.moment(), areg.kip * areg.ft)

    debug_callback(
        partial(plot_cb, version=version),
        positions=prof.positions,
        shear=convert_unit(prof.shear(), areg.kip),
        moment=mom,
    )

    max_mom = jnp.max(jnp.abs(mom))

    max_stress = 15 * areg.ksi

    min_modulus = convert_unit(max_mom / max_stress, areg.inch**3)

    debug_print("section modulus", min_modulus)

    if version == 2:
        flange_force = (max_mom / 2) / (48.0 * areg.inch / 2)
        debug_print("flange force per flange", convert_unit(flange_force, areg.kip))
        per_flange = convert_unit(flange_force / max_stress, areg.inch**2)
        tot = per_flange * 2
        debug_print("area per flange", per_flange)
        debug_print("area total", tot)
        prev_area = 16.7 * 1.68 * areg.inch**2
        debug_print("saved", (prev_area - per_flange) / prev_area)

    debug_print()
    return prof


@jaxtyped(typechecker=typechecker)
def plot_cb(
    version: int,
    positions: Float[Array, "p"],
    shear: Float[Array, "p"],
    moment: Float[Array, "p"],
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_title("shear")
    ax1.set_ylim(-200, 200)
    plot_unit(positions, shear, ax=ax1)

    ax2.set_title("moment")
    ax2.set_ylim(-2000, 2000)
    plot_unit(positions, moment, ax=ax2)

    fig.tight_layout()
    # fig.subplots_adjust(left=0.2)
    fig.savefig(out_dir / f"output{version}.png", dpi=300)
    plt.close(fig)
