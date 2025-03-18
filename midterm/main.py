import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import oryx
import sympy as s
import sympy2jax
from jax import Array, lax
from jax import tree_util as jtu
from jax._src.typing import ArrayLike
from jax.experimental.checkify import check, checkify
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pintax import Unit, areg, convert_unit, quantity, sync_units, unitify, ureg
from pintax._utils import pretty_print
from pintax.functions import lstsq

from lib.batched import (
    batched,
    batched_vmap,
    batched_zip,
    do_batch,
    tree_do_batch,
    unbatch,
)
from lib.beam import force_profile, force_profile_builder
from lib.checkify import checkify_simple
from lib.graph import graph_t, point, pointid
from lib.jax_utils import debug_print, flatten_handler
from lib.lstsq import flstsq, flstsq_checked
from lib.plot import plot_unit
from lib.utils import (
    blike,
    bval,
    cast_unchecked,
    debug_callback,
    fval,
    ival,
    jit,
    return_of_fake,
    unique,
)
from midterm.build_graph import build_graph

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

np.set_printoptions(precision=3, suppress=True)


def solve_forces(g: graph_t):
    connnection_forces = g._connections.map(lambda _: jnp.array(0.0) * areg.pound)

    def get_eqs(connnection_forces=connnection_forces):
        aggr = g.sum_annotations(
            g._points.map(lambda _: jnp.zeros(3) * areg.pound),
            g.forces_aggregate(connnection_forces, density=1.0 * areg.pound / areg.m),
        )
        aggr_filtered = batched_vmap(
            lambda p, f: lax.select(p.fixed, on_true=jnp.zeros_like(f), on_false=f),
            g._points,
            aggr,
        )
        return aggr_filtered

    pass


@unitify
@jit
def solve_forces_final():
    g = build_graph()

    connnection_forces = g._connections.map(lambda _: jnp.array(0.0) * areg.pound)

    def get_eqs(connnection_forces=connnection_forces):
        aggr = g.sum_annotations(
            g._points.map(lambda _: jnp.zeros(3) * areg.pound),
            g.forces_aggregate(connnection_forces, density=1.0 * areg.pound / areg.m),
        )
        aggr_filtered = batched_vmap(
            lambda p, f: lax.select(p.fixed, on_true=jnp.zeros_like(f), on_false=f),
            g._points,
            aggr,
        )
        return aggr_filtered

    return g, flstsq(get_eqs, connnection_forces)


@unitify
def do_plot(res_):
    res = cast_unchecked.from_fn(solve_forces_final)(res_)
    g, ans = res
    forces = ans.x
    forces_errors = ans.errors

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)

    # lim = 30
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)

    ax.set_xlim(-20, 10)
    ax.set_ylim(-10, 20)
    ax.set_zlim(-10, 20)

    f_max = jnp.max(jnp.abs(forces.unflatten()))
    print("f_max", f_max)
    f_max = 500 * areg.pound

    def _color(x: fval, end: Array):
        x = lax.min(x / f_max * 10, 1.0)
        return x * end + (1 - x) * jnp.array([1.0, 1.0, 1.0]) * 0.5

    colors = forces.map(
        lambda x: lax.select(
            x > 0,
            on_true=_color(x, jnp.array([0.0, 0.0, 1.0])),
            on_false=_color(-x, jnp.array([1.0, 0.0, 0.0])),
        )
    )
    linewidths = forces.map(lambda x: (jnp.abs(x) / f_max * 10 + 0.2))

    line_collection = Line3DCollection(
        (g.get_lines() / areg.m).tolist(),
        colors=colors.unflatten().tolist(),
        linewidths=linewidths.unflatten().tolist(),
    )
    ax.add_collection3d(line_collection)
    # ax.plot(xs, ys, zs)

    # fixed_points, ct = g._points.filter(lambda x: x.fixed)
    plot_errors = batched_zip(g._points, forces_errors)
    plot_errors, ct = plot_errors.filter_arr(
        plot_errors.tuple_map(lambda p, e: jnp.linalg.norm(e) > 0.5 * areg.pound)
    )

    def _plot_errors(x: point, e: Array):
        cd = x.coords / areg.m
        v = e / areg.pound
        return jnp.stack([cd, cd + v])

    print("count:", ct)
    plot_error_lines = plot_errors[: int(ct)].tuple_map(_plot_errors).unflatten()
    line_collection = Line3DCollection(
        plot_error_lines.tolist(),
        colors=(0.0, 1.0, 0.0),
        linewidths=1.0,
    )
    ax.add_collection3d(line_collection)

    # ax.scatter(
    #     plot_points_coords[:, 0].tolist(),
    #     plot_points_coords[:, 1].tolist(),
    #     plot_points_coords[:, 2].tolist(),  # type: ignore
    #     c="r",
    #     marker="o",
    #     s=plot_points_s.tolist(),
    # )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
    # print(ys)


@unitify
def do_plot_simple():
    g = build_graph()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)

    ax.set_xlim(-20, 10)
    ax.set_ylim(-10, 20)
    ax.set_zlim(-10, 20)

    line_collection = Line3DCollection(
        (g.get_lines() / areg.m).tolist(),
    )
    ax.add_collection3d(line_collection)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
