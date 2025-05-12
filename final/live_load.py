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
from lib.jax_utils import debug_print, oryx_var
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
    weight_c = areg.kpounds
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

    n_angle = 12

    outer_r = 178.712 * areg.ft / 2
    inner_r = 35.682 * areg.ft / 2

    outer_h = 30.0 * areg.ft
    inner_h = 10.0 * areg.ft

    angles = batched.create_array(jnp.linspace(0, 2 * math.pi, n_angle + 1)[:-1])

    def mk_default_connect(a: pointid, b: pointid):
        return connection(
            a,
            b,
            force_per_deform=1.0 * force_per_deform_c,
            density=0.0 * weight_c / areg.ft,
        )

    g, outer_ring = g.add_point_batched(
        angles.map(
            lambda a: jnp.array([outer_r * jnp.cos(a), outer_r * jnp.sin(a), outer_h])
        )
    )
    g, inner_ring = g.add_point_batched(
        angles.map(
            lambda a: jnp.array([inner_r * jnp.cos(a), inner_r * jnp.sin(a), inner_h])
        )
    )

    g = g.add_connection_batched(
        batched_zip(outer_ring, outer_ring.roll(1)).tuple_map(mk_default_connect)
    )
    g = g.add_connection_batched(
        batched_zip(inner_ring, inner_ring.roll(1)).tuple_map(mk_default_connect)
    )
    g = g.add_connection_batched(
        batched_zip(inner_ring, outer_ring).tuple_map(mk_default_connect)
    )

    ####################
    # supports
    ####################

    # column
    g, outer_down = g.add_point_batched(
        angles.map(
            lambda a: jnp.array([outer_r * jnp.cos(a), outer_r * jnp.sin(a), 0.0])
        )
    )
    g = g.add_connection_batched(
        batched_zip(outer_ring, outer_down).tuple_map(mk_default_connect)
    )
    g = g.add_external_force_batched(
        batched_zip(
            outer_down,
            batched.create(
                oryx_var(
                    f"outer_down",
                    tag_external_force,
                    jnp.zeros((n_angle, 3)) * weight_c,
                ),
                (n_angle,),
            ),
        ).tuple_map(force_annotation)
    )

    # tension
    for d in ["left", "right"]:
        d_mul = -1 if d == "left" else 1
        ang_bias = 5.0 * areg.degrees
        g, outer_slant = g.add_point_batched(
            angles.map(
                lambda a: jnp.array(
                    [
                        (outer_r + 10 * areg.ft) * jnp.cos(a + ang_bias * d_mul),
                        (outer_r + 10 * areg.ft) * jnp.sin(a + ang_bias * d_mul),
                        0.0,
                    ]
                )
            )
        )
        g = g.add_connection_batched(
            batched_zip(outer_ring, outer_slant).tuple_map(
                lambda a, b: connection(
                    a,
                    b,
                    tension_only=True,
                    force_per_deform=0.002 * force_per_deform_c,
                )
            )
        )
        g = g.add_external_force_batched(
            batched_zip(
                outer_slant,
                batched.create(
                    oryx_var(
                        f"outer_slant_{d}",
                        tag_external_force,
                        jnp.zeros((n_angle, 3)) * weight_c,
                    ),
                    (n_angle,),
                ),
            ).tuple_map(force_annotation)
        )

    # tension, center
    lean_center = inner_h * jnp.tan(45 * areg.degrees)
    g, inner_down = g.add_point_batched(
        angles.map(
            lambda a: jnp.array(
                [
                    (inner_r - lean_center) * jnp.cos(a),
                    (inner_r - lean_center) * jnp.sin(a),
                    0.0,
                ]
            )
        )
    )
    g = g.add_connection_batched(
        batched_zip(inner_ring, inner_down).tuple_map(
            lambda a, b: connection(
                a,
                b,
                tension_only=True,
                force_per_deform=1.0 * force_per_deform_c,
            )
        )
    )
    g = g.add_external_force_batched(
        batched_zip(
            inner_down,
            batched.create(
                oryx_var(
                    f"inner_down",
                    tag_external_force,
                    jnp.zeros((n_angle, 3)) * weight_c,
                ),
                (n_angle,),
            ),
        ).tuple_map(force_annotation)
    )

    ####################
    # loads
    ####################

    tot_live = 30 * areg.pound / areg.ft**2 * (outer_r**2) * math.pi / 2
    per_col = tot_live / 6 / 2
    debug_print("per_col", per_col)

    # for ring in [outer_ring, inner_ring]:
    for ring in [inner_ring]:
        g, viz_points = g.add_point_batched(
            ring.map(
                lambda p: g.get_point(p).coords + jnp.array([0.0, 0, 10.0]) * areg.ft
            )
        )
        g = g.add_connection_batched(
            batched_zip(ring, viz_points).tuple_map(mk_default_connect)
        )
        g = g.add_external_force_batched(
            viz_points.enumerate(
                lambda p, i: force_annotation(
                    p,
                    jnp.array([0.0, 0.0, tree_select(i < 6, -per_col, 0.0)]),
                    # jnp.array([0.0, 0.0, -100]) * weight_c,
                )
            )
        )

    return g
