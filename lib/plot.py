# pyright: reportUnusedCallResult=false
import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import oryx
from beartype import beartype as typechecker
from jax import Array, lax
from jax import tree_util as jtu
from jax.typing import ArrayLike
from jaxtyping import Float, jaxtyped
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pintax.oryx_ext as _
from lib.batched import batched, batched_vmap, batched_zip
from lib.displacement_based_forces import solve_forces
from lib.graph import connection, force_annotation, graph_t, point
from lib.jax_utils import fn_as_traced, oryx_unzip
from lib.lstsq import flstsq
from lib.utils import allow_autoreload, concatenate, fval, jit, tree_at_, vmap
from pintax import areg, quantity, unitify, ureg


@jaxtyped(typechecker=typechecker)
def plot_unit(
    x: Float[Array, "p"],
    y: Float[Array, "p"],
    ax: Axes | None = None,
):
    if ax is None:
        ax = plt.gca()

    x1 = quantity(x)
    y1 = quantity(y)

    y_min_v = float(jnp.min(y) / y1.u)
    y_max_v = float(jnp.max(y) / y1.u)

    x1_u = x1.u._pretty_print().format(use_color=False)
    y1_u = y1.u._pretty_print().format(use_color=False)

    ax.axhline(y=0, color="gray", linestyle="-")
    ax.axhline(y=y_min_v, color="red", linestyle=":")
    ax.axhline(y=y_max_v, color="red", linestyle=":")
    ax.plot(np.array(x1.m), np.array(y1.m), color="blue")
    ax.set_xlabel(x1_u)
    ax.set_ylabel(f"{y1_u} min={y_min_v:.1f} max={y_max_v:.1f}")

    # if file is not None:
    #     ax.subplots_adjust(left=0.2)
    #     ax.savefig(file, dpi=300)
    #     ax.clf()


class plot_graph_error_args(eqx.Module):
    forces_errors: batched[Array]
    allowed_error: Array


class plot_graph_args(eqx.Module):
    graph: graph_t
    connection_forces: batched[Array]

    err_args: plot_graph_error_args | None = None

    f_max: fval | None = None

    x_lim: tuple[float, float] = (0, 120)
    y_lim: tuple[float, float] = (0, 40)


def plot_graph_forces(arg: plot_graph_args):
    fig = plt.figure(figsize=(18, 8))
    # fig = plt.figure()
    # fig.suptitle(
    #     "redesign (forces are scaled down 2x when drawn, compared to previous plots)",
    #     fontsize=32,
    # )

    computed_zorder = True
    ax = fig.add_subplot(111, projection="3d", computed_zorder=computed_zorder)
    assert isinstance(ax, Axes3D)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_zlim(-5, 50)
    ax.set_aspect("equal")

    plot_graph_forces_ax(ax, arg)


def plot_graph_forces_ax(ax: Axes3D, arg: plot_graph_args):
    g = arg.graph
    forces = arg.connection_forces

    assert arg.err_args is None
    # forces_errors = arg.forces_errors

    ax.set_axis_off()
    # assert False

    # lim = 30
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)

    f_max = jnp.max(jnp.abs(forces.unflatten()))
    print("f_max", f_max)
    f_max = arg.f_max if arg.f_max is not None else f_max

    def _color_from_force(x: fval, _end: Array):
        x = lax.min(x / f_max * 10, 1.0)
        return x * _end + (1 - x) * jnp.array([1.0, 1.0, 1.0]) * 0.5

    def color_width_from_force(x: fval):
        color = lax.select(
            x > 0,
            on_true=_color_from_force(x, jnp.array([1.0, 0.0, 0.0])),
            on_false=_color_from_force(-x, jnp.array([0.0, 0.0, 1.0])),
        )
        # width = jnp.minimum(jnp.abs(x) / f_max, 1.0) * 10 + 0.2
        width = jnp.minimum(jnp.abs(x) / f_max, 1.0) * 10 + 1.0
        return color, width

    lines, (colors, linewidths) = (
        batched_zip(g._connections, forces)
        # .filter_concrete(lambda c_f: quantity(jnp.abs(c_f[1])).m_arr > 1e-5)
        .tuple_map(
            lambda c, f: (
                jnp.stack([g.get_point(c.a).coords, g.get_point(c.b).coords]),
                color_width_from_force(f),
            )
        ).unflatten()
    )
    line_collection = Line3DCollection(
        (lines / areg.ft).tolist(),
        colors=colors.tolist(),
        linewidths=linewidths.tolist(),
        # zorder=1,
    )
    ax.add_collection3d(line_collection)

    # def _plot_external_fs(x: point, f: Array):
    #     cd = x.coords / areg.ft
    #     f_n = jnp.linalg.norm(f)
    #     cd_other = cd - f / f_max * 20.0
    #     return jnp.stack([cd_other, cd]), color_width_from_force(f_n), cd_other

    # points_external_fs, _count = batched_zip(g._points, g.sum_annotations()).filter(
    #     lambda p_f: jnp.linalg.norm(p_f[1]) > 0.0
    # )
    # (
    #     external_fs_segs,
    #     (external_fs_colors, external_fs_linwidths),
    #     external_fs_points,
    # ) = (
    #     points_external_fs[: int(_count)].tuple_map(_plot_external_fs).unflatten()
    # )
    # ax.add_collection3d(
    #     Line3DCollection(
    #         external_fs_segs.tolist(),
    #         colors=external_fs_colors.tolist(),
    #         linewidths=external_fs_linwidths.tolist(),
    #     )
    # )
    # ax.scatter(
    #     external_fs_points[:, 0].tolist(),
    #     external_fs_points[:, 1].tolist(),
    #     external_fs_points[:, 2].tolist(),  # pyright: ignore[reportArgumentType]
    #     color=(0.0, 0.0, 0.0),
    #     marker="o",
    #     # s=(intensity * 20.0).tolist(),  # pyright: ignore[reportArgumentType]
    #     # s=plot_points_s.tolist(),
    #     # zorder=6,
    # )

    # ax.plot(xs, ys, zs)

    # # fixed_points, ct = g._points.filter(lambda x: x.fixed)
    # plot_errors = batched_zip(g._points, forces_errors)
    # plot_errors, ct = plot_errors.filter_arr(
    #     plot_errors.tuple_map(lambda p, e: jnp.linalg.norm(e) > 0.2 * areg.pound)
    # )

    # def _plot_errors(x: point, e: Array):
    #     cd = x.coords / areg.ft
    #     v = e / areg.pound * 2
    #     return jnp.stack([cd, cd + v]), cd + v, jnp.maximum(jnp.linalg.norm(v), 0.2)

    # print("count:", ct)
    # ct = int(ct)
    # if ct > 0:
    #     plot_error_lines, plot_error_points, intensity = (
    #         plot_errors[:ct].tuple_map(_plot_errors).unflatten()
    #     )
    #     line_collection = Line3DCollection(
    #         plot_error_lines.tolist(),
    #         colors=(0.0, 1.0, 0.0),
    #         linewidths=(intensity * 3.0).tolist(),
    #         zorder=5,
    #     )
    #     ax.add_collection3d(line_collection)

    #     ax.scatter(
    #         plot_error_points[:, 0].tolist(),
    #         plot_error_points[:, 1].tolist(),
    #         plot_error_points[:, 2].tolist(),  # pyright: ignore[reportArgumentType]
    #         color=(0.0, 0.0, 0.0),
    #         marker="o",
    #         s=(intensity * 20.0).tolist(),  # pyright: ignore[reportArgumentType]
    #         # s=plot_points_s.tolist(),
    #         zorder=6,
    #     )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_graph_forces2d(arg: plot_graph_args):
    g = arg.graph
    forces = arg.connection_forces

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(arg.x_lim[0], arg.x_lim[1])
    ax.set_ylim(arg.y_lim[0], arg.y_lim[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    f_max = jnp.max(jnp.abs(forces.unflatten()))
    print("f_max", f_max)
    f_max = arg.f_max if arg.f_max is not None else f_max

    def _color_from_force(x: fval, _end: Array):
        x = lax.min(x / f_max * 10, 1.0)
        return x * _end + (1 - x) * jnp.array([1.0, 1.0, 1.0]) * 0.5

    def color_width_from_force(x: fval):
        color = lax.select(
            x > 0,
            on_true=_color_from_force(x, jnp.array([1.0, 0.0, 0.0])),
            on_false=_color_from_force(-x, jnp.array([0.0, 0.0, 1.0])),
        )
        # width = jnp.minimum(jnp.abs(x) / f_max, 1.0) * 10 + 0.2
        width = jnp.minimum(jnp.abs(x) / f_max, 1.0) * 10 + 1.0
        return color, width

    lines, (colors, linewidths) = (
        batched_zip(g._connections, forces)
        # .filter_concrete(lambda c_f: quantity(jnp.abs(c_f[1])).m_arr > 1e-5)
        .tuple_map(
            lambda c, f: (
                jnp.stack([g.get_point(c.a).coords, g.get_point(c.b).coords]),
                color_width_from_force(f),
            )
        ).unflatten()
    )
    line_collection = LineCollection(
        (lines / areg.ft).tolist(),
        colors=colors.tolist(),
        linewidths=linewidths.tolist(),
        # zorder=1,
    )
    ax.add_collection(line_collection)

    # def _plot_external_fs(x: point, f: Array):
    #     cd = x.coords / areg.ft
    #     f_n = jnp.linalg.norm(f)
    #     cd_other = cd - f / f_max * 20.0
    #     return jnp.stack([cd_other, cd]), color_width_from_force(f_n), cd_other

    # (
    #     external_fs_segs,
    #     (external_fs_colors, external_fs_linwidths),
    #     external_fs_points,
    # ) = (
    #     batched_zip(g._points, g.sum_annotations())
    #     .filter_concrete(lambda p_f: jnp.linalg.norm(p_f[1]) > 0.0)
    #     .tuple_map(_plot_external_fs)
    #     .unflatten()
    # )
    # ax.add_collection(
    #     LineCollection(
    #         external_fs_segs.tolist(),
    #         colors=external_fs_colors.tolist(),
    #         linewidths=external_fs_linwidths.tolist(),
    #     )
    # )
    # ax.scatter(
    #     external_fs_points[:, 0].tolist(),
    #     external_fs_points[:, 1].tolist(),
    #     color=(0.0, 0.0, 0.0),
    #     marker="o",
    #     s=20,
    #     zorder=6,
    # )

    plt.show()
