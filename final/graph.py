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
from lib.graph import connection, force_annotation, graph_t
from lib.jax_utils import oryx_var
from lib.lstsq import flstsq
from lib.utils import allow_autoreload, concatenate, fval, jit, tree_at_, vmap
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

    g = graph_t.create(3)

    outer_r = 30.0 * areg.m
    inner_r = 10.0 * areg.m
    n_rings = 10
    n_angle = 30
    # n_rings = 3
    # n_angle = 4

    x_pos = oryx_var("x_pos", tag_pos, jnp.linspace(inner_r, outer_r, n_rings)[1:-1])
    x_pos = concatenate([[inner_r], x_pos, [outer_r]])

    # z_pos_defaults = jnp.linspace(-1.0, 0.0, n_rings)
    z_pos_defaults = jnp.linspace(0.2, 1.0, n_rings) ** 2

    # z_pos_defaults = jax.vmap(lambda x: ((x - 1) / 2 + 1 - 1 / x / 10))(
    #     jnp.linspace(0.2, 1.0, n_rings)
    # )

    z_pos_defaults -= z_pos_defaults[0]
    z_pos_defaults /= z_pos_defaults[-1]
    z_pos_defaults *= 15 * areg.m

    z_pos = oryx_var("z_pos", tag_pos, z_pos_defaults[:-1])
    z_pos = concatenate([z_pos, z_pos_defaults[-1:]])

    angles = batched.create(jnp.linspace(0, 2 * math.pi, n_angle + 1)[:-1], (n_angle,))

    point_choords = angles.map(
        lambda a: batched.create((x_pos, z_pos), (len(x_pos),)).tuple_map(
            lambda x, z: jnp.array([x * jnp.cos(a), x * jnp.sin(a), z])
        )
    )
    g, points = g.add_point_batched(point_choords.uf)
    assert points.batch_dims() == (n_angle, n_rings)

    ring_connection_areas = oryx_var(
        "ring_connection_areas",
        tag_area,
        jnp.ones(n_rings),
    )
    ring_connections = vmap(
        (points.transpose([1, 0]), ring_connection_areas),
        lambda ring, area: batched_zip(ring, ring.roll(1)).tuple_map(
            lambda x, y: connection(
                x,
                y,
                force_per_deform=area * force_per_deform_c,
                # weight=area * weight_c,
                density=area * weight_c / areg.m,
            )
        ),
    )
    g = g._add_connection_batched(ring_connections)

    x_connection_areas = oryx_var(
        "x_connection_areas",
        tag_area,
        jnp.ones(n_rings - 1),
    )
    x_connections = points.map1d(
        lambda line: batched_zip(
            line[:-1], line[1:], batched.create(x_connection_areas, (n_rings - 1,))
        ).tuple_map(
            lambda x, y, a: connection(
                x,
                y,
                force_per_deform=a * force_per_deform_c,
                # weight=a * weight_c,
                density=a * weight_c / areg.m,
            )
        )
    )
    g = g._add_connection_batched(x_connections)

    outer_support_z = oryx_var("outer_support_z", tag_external_force, 0.0 * weight_c)
    outer_ring = points[:, -1]
    # outer_ring = points[:, -3]
    # outer_ring = points[:, 0]

    g = g.add_external_force_batched(
        outer_ring.map(
            lambda p: force_annotation(p, jnp.array([0.0, 0.0, outer_support_z]))
        )
    )

    return g
