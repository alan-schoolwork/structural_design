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

    def mk_default_connect(a: pointid, b: pointid):
        return connection(
            a,
            b,
            force_per_deform=1.0 * force_per_deform_c,
            density=0.0 * weight_c / areg.ft,
        )

    outer_r = 178.712 * areg.ft / 2
    inner_r = 35.682 * areg.ft / 2

    n = 12 + 1

    x_pos = jnp.linspace(outer_r, inner_r, n)

    z_offset_by_x = jnp.linspace(30.0, 10.0, n) * areg.ft

    cs = jnp.cumsum(jnp.cumsum(jnp.linspace(1.0, inner_r / outer_r, n)))
    cs = cs - jnp.linspace(cs[0], cs[-1], len(cs))
    cs /= 3
    z_offset_by_x_under_additional = oryx_var(
        "z_offset_by_x_under_additional",
        tag_external_force,
        cs * areg.ft,
        store_scale=2000000,
    )

    g, top_row = g.add_point_batched(
        batched.create((x_pos, z_offset_by_x), (n,)).map(jnp.array)
    )

    g, bot_row = g.add_point_batched(
        batched_zip(
            top_row,
            batched.create_array(z_offset_by_x_under_additional),
        )[
            1:-1
        ].tuple_map(lambda pid, z: (g.get_point(pid).coords + jnp.array([0, z])))
    )

    g = g.add_connection_batched(
        batched_zip(top_row[1:], top_row[:-1]).tuple_map(
            lambda a, b: connection(
                a,
                b,
                force_per_deform=1.0 * force_per_deform_c,
            )
        )
    )
    g = g.add_connection_batched(
        batched_zip(
            batched.concat([top_row[:1], bot_row]),
            batched.concat([bot_row, top_row[-1:]]),
        ).tuple_map(
            lambda a, b: connection(
                a,
                b,
                force_per_deform=1.0 * force_per_deform_c,
            )
        )
    )
    g = g.add_connection_batched(
        batched_zip(top_row[1:-1], bot_row).tuple_map(
            lambda a, b: connection(
                a,
                b,
                force_per_deform=1.0 * force_per_deform_c,
            )
        )
    )

    g, top_forces_points = g.add_point_batched(
        top_row.map(
            lambda pid: (g.get_point(pid).coords + jnp.array([0, 5.0 * areg.ft]))
        )
    )
    g = g.add_connection_batched(
        batched_zip(top_row, top_forces_points).tuple_map(mk_default_connect)
    )
    g = g.add_external_force_batched(
        top_forces_points.map(
            lambda p: force_annotation(
                p,
                jnp.array([0, -g.get_point(p).coords[0] * weight_c / areg.m]),
            ),
        )
    )

    def add_external(g: graph_t, p: pointid, direction: Array, f: Array):
        g, p_dir = g.add_point(g.get_point(p).coords + direction)
        g = g.add_connection_batched(batched.create(mk_default_connect(p, p_dir)))
        g = g.add_external_force(force_annotation(p_dir, f))
        return g

    g = add_external(
        g,
        top_row[0].unwrap(),
        jnp.array([10.0 * areg.ft, 0.0]),
        jnp.array([oryx_var("outer_support_x", tag_external_force, weight_c), 0.0]),
    )
    g = add_external(
        g,
        top_row[0].unwrap(),
        jnp.array([0.0, -30.0 * areg.ft]),
        jnp.array([0.0, oryx_var("outer_support_z", tag_external_force, weight_c)]),
    )
    g = add_external(
        g,
        top_row[-1].unwrap(),
        jnp.array([-10.0 * areg.ft, 0.0]),
        jnp.array([oryx_var("inner_tension_x", tag_external_force, weight_c), 0.0]),
    )

    return g
