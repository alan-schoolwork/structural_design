from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, lax
from matplotlib.collections import LineCollection

from pintax import areg, quantity, unitify
from pintax._helpers import zeros_like_same_unit
from pintax.unstable import pint_registry

from .batched import batched, batched_vmap, batched_zip
from .graph import connection, force_annotation, graph_t
from .jax_utils import debug_print, fn_as_traced
from .lstsq import flstsq, flstsq_r
from .utils import (
    allow_autoreload,
    cast,
    cast_unchecked,
    fval,
    jit,
    pformat_repr,
    tree_at_,
    tree_map,
)


class solve_forces_r[T](eqx.Module):
    flstsq_out: flstsq_r[tuple[batched[Array], T], batched[Array]]

    graph_args: T
    graph: graph_t

    connection_forces: batched[Array]

    point_force_errors: batched[Array]

    __repr__ = pformat_repr


def aggregate_all_forces(
    g: graph_t, connection_forces: batched[Array]
) -> batched[Array]:

    def inner(c: connection, connection_force: Array):
        a = g.get_point(c.a)
        b = g.get_point(c.b)

        v = b.coords - a.coords
        v_len = jnp.linalg.norm(v)
        v_dir = v / v_len  # vector a->b

        weight = c.weight + c.density * v_len
        half_w = jnp.array([0, 0, weight]) / 2

        ans1 = batched.create(force_annotation(c.a, -connection_force * v_dir - half_w))
        ans2 = batched.create(force_annotation(c.b, connection_force * v_dir - half_w))

        return batched.stack([ans1, ans2])

    annos = batched_zip(g._connections, connection_forces).tuple_map(inner)
    g = g.add_external_force_batched(annos.uf)
    aggr = g.sum_annotations()

    return aggr


def solve_forces[T](
    graph_fn: Callable[[T], graph_t], graph_ex: graph_t, arg_ex: T
) -> solve_forces_r[T]:

    def get_equations(solve_vars: tuple[batched[Array], T]):
        connection_forces_var, arg = solve_vars
        graph = graph_fn(arg)
        connection_forces = connection_forces_var.map(
            lambda x: x * quantity(graph._connections.get_arr(lambda me: me.weight)).u
        )
        equations = aggregate_all_forces(graph, connection_forces)
        return equations, graph, connection_forces

    f_args = (
        graph_ex._connections.map(lambda x: 0.0 * areg.dimensionless),
        arg_ex,
    )

    ans = flstsq(lambda x: get_equations(x)[0], f_args)

    debug_print("solve_forces: residuals=", ans.linear_residuals)

    _, graph, connection_forces = get_equations(ans.x)

    _, graph_args = ans.x

    return solve_forces_r(
        flstsq_out=ans,
        graph_args=graph_args,
        graph=graph,
        connection_forces=connection_forces,
        point_force_errors=ans.errors,
    )
