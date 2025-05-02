import math

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import oryx
from jax import Array, lax
from jax.typing import ArrayLike
from matplotlib.collections import LineCollection

import pintax.oryx_ext as _
from lib.batched import batched, batched_vmap, batched_zip
from lib.graph import connection, force_annotation, graph_t, pointid
from lib.jax_utils import oryx_var
from lib.lstsq import flstsq
from lib.utils import (
    allow_autoreload,
    concatenate,
    fval,
    jit,
    tree_at_,
    tree_select,
    vmap,
)
from pintax import areg, convert_unit, unitify
from pintax.unstable import pint_registry

tag_pos = "oryx_tag_position"
tag_area = "oryx_tag_area"
tag_external_force = "oryx_tag_external_force"

pint_registry().define("force_per_deform_c=[force_per_deform_c]")
pint_registry().define("weight_c=[weight_c]")


def build_graph() -> graph_t:

    print("build_graph: tracing")

    # force_per_deform_c = areg.force_per_deform_c
    weight_c = areg.weight_c
    force_per_deform_c = weight_c

    g = graph_t.create(
        3,
        connection_ex=connection(
            pointid.ex(),
            pointid.ex(),
            force_per_deform=0.0 * force_per_deform_c,
            weight=0.0 * weight_c,
            density=0.0 * weight_c / areg.ft,
        ),
    )

    outer_r = 110.0 * areg.ft
    inner_r = 10.0 * areg.ft

    n_sectors = 10
    n_angle_per_sector = 4
    n_angle = n_sectors * n_angle_per_sector

    n = 12 + 1

    x_pos = jnp.linspace(outer_r, inner_r, n)
    angles = jnp.linspace(0, 2 * math.pi, n_angle + 1)[:-1]

    # mid_z_offset_a = oryx_var(
    #     "mid_z_offset_a", tag_external_force, 4.0, store_scale=1.0
    # )

    z_offset_by_x = jnp.linspace(0.0, -15.0, n) * areg.ft

    cs = jnp.cumsum(jnp.cumsum(jnp.linspace(1.0, 0.0, n)))
    cs = cs - jnp.linspace(cs[0], cs[-1], len(cs))
    z_offset_by_x_under_additional = cs * areg.ft

    z_factor_x = jnp.linspace(1.0, 0.2, n)
    z_offset_by_a = jnp.array([0.0, 10.0, 12.9, 10.0] * n_sectors) * areg.ft

    point_coords = (
        batched.create((x_pos, z_offset_by_x, z_factor_x), (n,))
        .tuple_map(
            lambda x, z1, zf: batched.create(
                (angles, z_offset_by_a), (n_angle,)
            ).tuple_map(
                lambda a, z2: jnp.array([x * jnp.cos(a), x * jnp.sin(a), z1 + z2 * zf]),
            ),
        )
        .uf
    )
    # point_coords = batched.create(
    #     oryx_var("all_points", tag_external_force, point_coords.arr, store_scale=100.0),
    #     point_coords.batch_dims(),
    # )
    g, points = g.add_point_batched(point_coords)
    assert points.batch_dims() == (n, n_angle)

    # x connections
    g = g.add_connection_batched(
        points.transpose()
        .enumerate1d(
            lambda i, p: batched_zip(p[1:], p[:-1]).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    force_per_deform=lax.select(
                        i % n_angle_per_sector == 0,
                        on_true=5.0 * force_per_deform_c,
                        on_false=0.1 * force_per_deform_c,
                    ),
                    density=0.0 * weight_c / areg.m,
                    # compression_only=i % n_angle_per_sector != 0,
                )
            )
        )
        .uf
    )

    # ring connections
    g = g.add_connection_batched(
        points.map1d(
            lambda p: batched_zip(p, p.roll(1)).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    force_per_deform=0.1 * force_per_deform_c,
                    density=1.0 * weight_c / areg.m,
                    compression_only=True,
                )
            )
        ).uf
    )

    # diagonal
    g = g.add_connection_batched(
        batched.arange(n - 1)
        .map(
            lambda x_idx: batched.concat(
                [
                    batched_zip(points[x_idx], points[x_idx + 1].roll(1)),
                    batched_zip(points[x_idx], points[x_idx + 1].roll(-1)),
                ]
            ).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    force_per_deform=0.1 * force_per_deform_c,
                    density=0.0 * weight_c / areg.m,
                    compression_only=True,
                )
            )
        )
        .uf
    )

    points_sectors = points.reshape(n, n_sectors, n_angle_per_sector)
    points_main = points_sectors[:, :, 0]

    # points on the tension
    point_coords_under = batched_zip(
        points_main,
        batched.create(
            z_offset_by_x_under_additional[:, np.newaxis],
            points_main.batch_dims(),
            broadcast=True,
        ),
    ).tuple_map(lambda pid, z: (g.get_point(pid).coords + jnp.array([0, 0, z])))
    g, points_under = g.add_point_batched(point_coords_under)

    g = g.add_connection_batched(
        points_under.map1d(
            lambda p: batched_zip(p[1:], p[:-1]).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    5.0 * force_per_deform_c,
                    density=0.0 * weight_c / areg.m,
                    # tension_only=True,
                )
            ),
            in_axes=1,
        ).uf
    )

    # tensions up
    g = g.add_connection_batched(
        batched_zip(points_under[1:-1, :], points_main[1:-1, :]).tuple_map(
            lambda a, b: connection(
                a,
                b,
                5.0 * force_per_deform_c,
                density=0.0 * weight_c / areg.m,
            )
        )
    )
    # tensions diag
    g = g.add_connection_batched(
        batched_zip(points_under[1:-2, :], points_main[2:-1, :]).tuple_map(
            lambda a, b: connection(
                a,
                b,
                0.1 * force_per_deform_c,
                density=0.0 * weight_c / areg.m,
                tension_only=True,
            )
        )
    )
    # tensions diag
    g = g.add_connection_batched(
        batched_zip(points_under[2:-1, :], points_main[1:-2, :]).tuple_map(
            lambda a, b: connection(
                a,
                b,
                0.1 * force_per_deform_c,
                density=0.0 * weight_c / areg.m,
                tension_only=True,
            )
        )
    )

    # ring connections on main_lines
    g = g.add_connection_batched(
        # points_sectors[jnp.array([0, -1]), :, 0]
        points_sectors[:, :, 0]
        .enumerate1d(
            lambda i, p: batched_zip(p, p.roll(1)).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    force_per_deform=lax.select(
                        (i == 0) | (i == n - 1),
                        on_true=5.0 * force_per_deform_c,
                        on_false=0.05 * force_per_deform_c,
                    ),
                    density=0.0 * weight_c / areg.m,
                    tension_only=(i != 0),
                )
            )
        )
        .uf
    )

    # g = g.add_connection_batched(
    #     batched_zip(
    #         points[0],
    #         points[0].roll(2),
    #     ).tuple_map(
    #         lambda a, b: connection(
    #             a,
    #             b,
    #             force_per_deform=0.1 * force_per_deform_c,
    #             density=0.0 * weight_c / areg.m,
    #         )
    #     )
    # )

    # # test
    # for i in [1, 2, 3]:
    #     g = g.add_connection_batched(
    #         batched_zip(
    #             points_sectors[0, :, i - 1],
    #             points_sectors[0, :, i + 1],
    #         ).tuple_map(
    #             lambda a, b: connection(
    #                 a,
    #                 b,
    #                 force_per_deform=0.1 * force_per_deform_c,
    #                 density=0.0 * weight_c / areg.m,
    #             )
    #         )
    #     )

    # test
    for i in [0, 1]:
        for j in [1, 2, 3]:
            g = g.add_connection_batched(
                batched.concat(
                    [
                        batched_zip(
                            points_sectors[0, :, j],
                            points_sectors[i, :, 0],
                        ),
                        batched_zip(
                            points_sectors[0, :, j],
                            points_sectors[i, :, 0].roll(-1),
                        ),
                    ]
                ).tuple_map(
                    lambda a, b: connection(
                        a,
                        b,
                        force_per_deform=0.5 * force_per_deform_c,
                        density=0.0 * weight_c / areg.m,
                    )
                )
            )

    # test2
    for i in [1, 2, 3]:
        g = g.add_connection_batched(
            batched_zip(
                points_sectors[-1, :, i],
                points_sectors[-1, :, i].roll(1),
            ).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    force_per_deform=1.0 * force_per_deform_c,
                    density=0.0 * weight_c / areg.m,
                )
            )
        )

    outer_ring = points[0]
    support_points = outer_ring.reshape(n_sectors, -1)[:, 0]
    assert support_points.batch_dims() == (n_sectors,)

    # support_points = points[0]

    # print("support_points", support_points)
    # assert False

    outer_support_z = oryx_var(
        "outer_support_z", tag_external_force, jnp.zeros(len(support_points)) * weight_c
    )
    g = g.add_external_force_batched(
        batched_zip(
            support_points, batched.create(outer_support_z, (len(outer_support_z),))
        ).tuple_map(
            lambda p, f: force_annotation(
                p,
                jnp.array([0.0, 0.0, f]),
            ),
        )
    )

    # g = g.add_external_force_batched(
    #     points[5:10, 32:37].map(
    #         lambda p: force_annotation(
    #             p,
    #             jnp.array([0.0, 0.0, -5.0]) * areg.weight_c,
    #         )
    #     ),
    # )

    return g
