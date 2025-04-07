import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import oryx
import sympy as s
import sympy2jax
from jax import Array, lax
from jax._src.typing import ArrayLike
from matplotlib.collections import LineCollection

from lib.batched import batched, batched_vmap, batched_zip, tree_do_batch
from lib.graph import connection, force_annotation, graph_t, point, pointid
from lib.lstsq import flstsq
from lib.utils import allow_autoreload, blike, fval, jit, tree_at_, unique
from pintax import areg, sync_units, unitify

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

np.set_printoptions(precision=3, suppress=True)


def build_graph() -> tuple[graph_t, batched[force_annotation]]:
    g = graph_t.create(2)

    top_coords = (
        jnp.array(
            [
                [-12.0, 0.0],
                [-6.0, 2.0],
                [0.0, 4.0],
                [6.0, 2.0],
                [12.0, 0.0],
            ]
        )
        * areg.inch
    )

    bot_coords = jnp.array([-8.0, -4.0, 0.0, 4.0, 8.0]) * areg.inch

    g, bot_pts = g.add_point_batched(
        batched.create(bot_coords, (len(bot_coords),)).map(
            lambda x: jnp.array([x, 0.0])
        )
    )
    g, top_pts = g.add_point_batched(batched.create(top_coords, (len(top_coords),)))

    bot_pts = batched.concat([top_pts[:1], bot_pts, top_pts[-1:]])

    g = g.add_connection_batched(bot_pts[1:], bot_pts[:-1])
    g = g.add_connection_batched(top_pts[1:], top_pts[:-1])

    g = g.add_connection_batched(top_pts[1], bot_pts[2])
    g = g.add_connection_batched(top_pts[-2], bot_pts[-3])

    # g = g.add_connection_batched(top_pts, bot_pts[1:])
    # g = g.add_connection_batched(top_pts, bot_pts[:-1])

    hold = 100.0 * areg.force_pounds
    top_contact = top_pts[
        jnp.array(
            [len(top_pts) // 2 - 1, len(top_pts) // 2],
        )
    ]
    forces1 = top_contact.map(
        lambda x: force_annotation(x, jnp.array([0.0, -hold / len(top_contact)])),
    )

    bot_contact = bot_pts[jnp.array([0, -1])]
    forces2 = bot_contact.map(
        lambda x: force_annotation(x, jnp.array([0.0, hold / len(bot_contact)])),
    )

    return g, batched.concat([forces1, forces2])


def do_plot_simple():
    g, forces = build_graph()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    line_collection = LineCollection(
        (g.get_lines() / areg.inch).tolist(),
        linewidths=1.0,
    )
    ax.add_collection(line_collection)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()
    # plt.tight_layout()
    # plt.savefig("structure.png", dpi=300)
    # plt.close()


@jit
def solve_forces(g: graph_t, external_forces: batched[force_annotation]):

    connnection_forces = g._connections.map(lambda _: 0.0 * areg.force_pounds)

    def get_eqs(connnection_forces: batched[Array]):

        forces = batched.concat(
            [external_forces.reshape(-1), g.forces_aggregate(connnection_forces)]
        )

        aggr = g.sum_annotations(g._points.map(lambda _: jnp.zeros((2,))), forces)
        aggr_filtered = batched_vmap(
            lambda p, f: lax.select(
                p.accepts_force, on_true=jnp.zeros_like(f), on_false=f
            ),
            g._points,
            aggr,
        )
        return aggr_filtered

    return flstsq(get_eqs, connnection_forces)


# @jit
def displacement_based_forces_solve(
    g: graph_t, external_forces: batched[force_annotation]
):

    def forces_from_coords(
        coords: batched[Array],
    ) -> tuple[batched[Array], batched[Array]]:
        g2 = tree_at_(lambda g: g._points.unflatten().coords, g, coords.unflatten())

        annos, connection_fs = g2.displacement_based_forces(1.0 * areg.force_pound)

        forces = batched.concat([external_forces.reshape(-1), annos])

        aggr = g.sum_annotations(g._points.map(lambda _: jnp.zeros((2,))), forces)
        return aggr, connection_fs

    def forces_from_coords_tangents(
        tangents: batched[Array],
    ) -> tuple[batched[Array], batched[Array]]:

        primals = g._points.map(lambda x: x.coords)
        (primals_out, _), (tangents_out, connection_fs) = jax.jvp(
            forces_from_coords, (primals,), (tangents,)
        )

        return (
            batched_vmap(lambda x, y: x + y, primals_out, tangents_out),
            connection_fs,
        )

    ans = flstsq(
        lambda x: forces_from_coords_tangents(x)[0],
        g._points.map(lambda x: jnp.zeros_like(x.coords) * areg.inch),
    )

    connection_forces = forces_from_coords_tangents(ans.x)[1]

    return ans, connection_forces


@allow_autoreload
@unitify
def main():

    g, ext_forces = build_graph()
    ans, forces = displacement_based_forces_solve(g, ext_forces)

    print(ans.x)
    print(forces)
    # print("connection_fs")
    # print(connection_fs)
    # return ans

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    f_max = jnp.max(jnp.abs(forces.unflatten()))
    print("f_max", f_max)

    def _color(x: fval, end: Array):
        x = lax.min(x / f_max * 10, 1.0)
        return x * end + (1 - x) * jnp.array([1.0, 1.0, 1.0]) * 0.5

    colors = forces.map(
        lambda x: lax.select(
            x > 0,
            on_true=_color(x, jnp.array([1.0, 0.0, 0.0])),
            on_false=_color(-x, jnp.array([0.0, 0.0, 1.0])),
        )
    )
    linewidths = forces.map(lambda x: (jnp.abs(x) / f_max * 10 + 0.2))

    line_collection = LineCollection(
        (g.get_lines() / areg.inch).tolist(),
        colors=colors.unflatten().tolist(),
        linewidths=linewidths.unflatten().tolist(),
    )
    ax.add_collection(line_collection)

    plt.show()
