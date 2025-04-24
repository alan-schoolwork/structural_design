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
            density=0.0 * weight_c / areg.m,
        ),
    )

    outer_r = 30.0 * areg.m
    inner_r = 5.0 * areg.m
    n_angle = 30

    n = 10
    # n = 3
    # n = 4

    x_pos = oryx_var("x_pos", tag_pos, jnp.linspace(inner_r, outer_r, n)[1:-1])
    x_pos = concatenate([[inner_r], x_pos, [outer_r]])

    z_pos_defaults = jnp.linspace(-1.0, 0.0, n)
    z_max = 5 * areg.m

    # z_pos_defaults = jnp.linspace(0.2, 1.0, n) ** 2
    # z_max = 5 * areg.m

    # z_pos_defaults = jax.vmap(lambda x: ((x - 1) / 2 + 1 - 1 / x / 10))(
    #     jnp.linspace(0.2, 1.0, n)
    # )
    # z_max = 10 * areg.m

    # z_pos_defaults = jax.vmap(lambda x: (1 - 1 / x / 10))(jnp.linspace(0.2, 1.0, n))
    # z_max = 10 * areg.m

    z_pos_defaults -= z_pos_defaults[0]
    z_pos_defaults /= z_pos_defaults[-1]
    z_pos_defaults *= z_max

    z_pos = oryx_var("z_pos", tag_pos, z_pos_defaults[:-1])
    z_pos = concatenate([z_pos, z_pos_defaults[-1:]])

    angles = batched.create(jnp.linspace(0, 2 * math.pi, n_angle + 1)[:-1], (n_angle,))

    def per_angle(a: fval, unbatch_dims: tuple[int, ...]) -> batched[pointid]:
        nonlocal g

        def add_point(x: fval, z: fval, unbatch_dims2: tuple[int, ...]) -> pointid:
            nonlocal g
            choord = jnp.array([x * jnp.cos(a), x * jnp.sin(a), z])
            g, ans = g.add_point(choord, (*unbatch_dims, *unbatch_dims2))
            return ans

        def add_connection(c: connection, unbatch_dims2: tuple[int, ...]):
            nonlocal g
            g = g.add_connection_unbatched(c, (*unbatch_dims, *unbatch_dims2))

        top_row = batched.create((x_pos, z_pos), (n,)).tuple_map(
            lambda x, z: add_point(x, z, (n,))
        )
        bot_row = batched.create(
            (
                (x_pos[1:] + x_pos[:-1]) / 2,
                (z_pos[1:] + z_pos[:-1]) / 2 - 1.0 * areg.m,
            ),
            (n - 1,),
        ).tuple_map(lambda x, z: add_point(x, z, (n - 1,)))

        cs = batched.concat(
            [
                batched_zip(top_row[1:], top_row[:-1]).tuple_map(
                    lambda a, b: (a, b, 1.0)
                ),
                #
                batched_zip(bot_row[1:], bot_row[:-1]).tuple_map(
                    lambda a, b: (a, b, 1.0)
                    # lambda a, b: (a, b, 0.1)
                ),
                #
                batched_zip(top_row[1:], bot_row).tuple_map(
                    lambda a, b: (a, b, 1.0),
                    # lambda a, b: (a, b, 0.0),
                ),
                batched_zip(top_row[:-1], bot_row).tuple_map(
                    lambda a, b: (a, b, 1.0),
                    # lambda a, b: (a, b, 0.0),
                ),
            ]
        )
        cs.tuple_map(
            lambda a, b, fpd: add_connection(
                connection(
                    a,
                    b,
                    force_per_deform=fpd * force_per_deform_c,
                    density=0.0 * weight_c / areg.m,
                ),
                (len(cs),),
            ),
        )
        return batched.concat([top_row, bot_row])

    all_points = angles.map(lambda a: per_angle(a, (len(angles),))).uf
    assert all_points.batch_dims() == (n_angle, (2 * n - 1))

    # test_area = oryx_var(
    #     "test_area", tag_external_force, 0.2 * force_per_deform_c, store_scale=10000
    # )

    # ring_connection_areas = oryx_var(
    #     "ring_connection_areas",
    #     tag_area,
    #     jnp.ones(2 * n - 1),
    # )
    ring_connections = vmap(
        # (all_points[:, jnp.array([0, n - 1])].transpose([1, 0]),),
        (all_points.transpose([1, 0]), jnp.arange(all_points.batch_dims()[1])),
        lambda ring, i: batched_zip(ring, ring.roll(1)).tuple_map(
            lambda x, y: connection(
                x,
                y,
                # force_per_deform=1.0 * force_per_deform_c,
                force_per_deform=tree_select(
                    (i == n - 1) | (i == 0),
                    on_true=5.0 * force_per_deform_c,
                    # on_false=0.0 * force_per_deform_c,
                    # on_false=0.2 * force_per_deform_c,
                    on_false=0.5 * force_per_deform_c,
                    # on_false=1.0 * force_per_deform_c,
                ),
                # weight=area * weight_c,
                density=1.0 * weight_c / areg.m,
            )
        ),
    )
    g = g.add_connection_batched(ring_connections)

    outer_support_z = oryx_var("outer_support_z", tag_external_force, 0.0 * weight_c)
    outer_support_x = oryx_var(
        "outer_support_x", tag_external_force, 0.0 * weight_c, store_scale=1000.0
    )
    outer_ring = all_points[:, n - 1]
    # outer_ring = all_points[:, n - 2]
    # outer_ring = points[:, -3]
    # outer_ring = points[:, 0]

    g = g.add_external_force_batched(
        batched_zip(outer_ring, angles).enumerate(
            lambda p_a, i: force_annotation(
                # p, jnp.array([0.0, 0.0, tree_select(i % 2 == 0, outer_support_z, 0.0)])
                p_a[0],
                jnp.array(
                    [
                        outer_support_x * jnp.cos(p_a[1]),
                        outer_support_x * jnp.sin(p_a[1]),
                        outer_support_z,
                    ]
                ),
            )
        )
    )

    # inner_ring = all_points[:, 0]
    # inner_support_z = oryx_var(
    #     "inner_support_z",
    #     tag_external_force,
    #     0.0 * weight_c,
    #     # store_scale=5000.0,
    # )
    # g = g.add_external_force_batched(
    #     inner_ring.enumerate(
    #         lambda p, i: force_annotation(p, jnp.array([0.0, 0.0, inner_support_z]))
    #     )
    # )

    return g
