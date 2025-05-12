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
        2,
        connection_ex=connection(
            pointid.ex(),
            pointid.ex(),
            force_per_deform=0.0 * force_per_deform_c,
            weight=0.0 * weight_c,
            density=0.0 * weight_c / areg.ft,
        ),
    )

    outer_r = 178.712 * areg.ft / 2
    inner_r = 35.682 * areg.ft / 2

    outer_h = 30.0 * areg.ft
    inner_h = 10.0 * areg.ft

    g, points = g.add_point_batched(
        batched.create(
            jnp.array(
                [
                    [-outer_r, outer_h],
                    [-inner_r, inner_h],
                    [inner_r, inner_h],
                    [outer_r, outer_h],
                ]
            ),
            (4,),
        )
    )

    g = g.add_connection_batched(
        batched_zip(points[1:], points[:-1]).tuple_map(
            lambda a, b: connection(
                a,
                b,
                force_per_deform=1.0 * force_per_deform_c,
            )
        )
    )

    def add_external(
        g: graph_t, p: pointid, direction: Array, f: Array, tension_only: bool = False
    ):
        g, p_dir = g.add_point(g.get_point(p).coords + direction)
        g = g.add_connection_batched(
            batched.create(
                connection(
                    p,
                    p_dir,
                    tension_only=tension_only,
                    force_per_deform=1.0 * force_per_deform_c,
                    density=0.0 * weight_c / areg.ft,
                )
            )
        )
        g = g.add_external_force(force_annotation(p_dir, f))
        return g

    ####################
    # supports
    ####################

    # column
    for i in [0, 3]:
        g = add_external(
            g,
            points[i].unwrap(),
            jnp.array([0.0, -outer_h]),
            jnp.array(
                [0.0, oryx_var(f"outer_support_z_{i}", tag_external_force, weight_c)]
            ),
        )

    # tension
    for i in [0, 3]:
        g = add_external(
            g,
            points[i].unwrap(),
            jnp.array([(-10 if i == 0 else 10) * areg.ft, -outer_h]),
            oryx_var(
                f"outer_tension_{i}",
                tag_external_force,
                jnp.array([0.0 * weight_c, 0.0 * weight_c]),
            ),
            tension_only=True,
        )

    ####################
    # loads
    ####################

    fixed_load = 200 * weight_c
    live_load = 50 * weight_c

    for i, f in enumerate(
        [
            fixed_load / 2,
            fixed_load / 2,
            (fixed_load + live_load) / 2,
            (fixed_load + live_load) / 2,
        ]
    ):
        g = add_external(
            g,
            points[i].unwrap(),
            jnp.array([0.0, 5.0 * areg.ft]),
            jnp.array([0.0, -f]),
        )

    ####################

    return g
