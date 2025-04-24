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
from pintax import areg, unitify
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
    mid_z_offset_a = 3.9

    z_offset_by_x = jnp.linspace(0.0, -5, n) * areg.m
    z_factor_x = jnp.linspace(1.0, 0.1, n)
    z_offset_by_a = jnp.array([0.0, 3.0, mid_z_offset_a, 3.0] * n_sectors) * areg.m

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
                        on_false=1.0 * force_per_deform_c,
                    ),
                    density=0.0 * weight_c / areg.m,
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
                    force_per_deform=1.0 * force_per_deform_c,
                    density=1.0 * weight_c / areg.m,
                )
            )
        ).uf
    )

    # test?
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
                    force_per_deform=1.0 * force_per_deform_c,
                    density=0.0 * weight_c / areg.m,
                    # compression_only=True,
                )
            )
        )
        .uf
    )

    main_lines = points.reshape(
        n,
        n_sectors,
        n_angle_per_sector,
    )[:, :, 0]
    assert main_lines.batch_dims() == (n, n_sectors)

    # ring connections on main_lines
    g = g.add_connection_batched(
        main_lines.enumerate1d(
            lambda i, p: batched_zip(p, p.roll(1)).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    force_per_deform=5.0 * force_per_deform_c,
                    density=0.0 * weight_c / areg.m,
                    tension_only=(i != 0),
                )
            )
        ).uf
    )

    outer_ring = points[0]
    support_points = outer_ring.reshape(n_sectors, -1)[:, 0]
    assert support_points.batch_dims() == (n_sectors,)

    support_points = points[0]

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

    return g
