from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, lax
from matplotlib.collections import LineCollection

from pintax import areg, unitify
from pintax._helpers import zeros_like_same_unit
from pintax.unstable import pint_registry

from .batched import batched, batched_vmap
from .graph import connection, force_annotation, graph_t
from .jax_utils import debug_print, fn_as_traced, primal_to_zero
from .lstsq import flstsq
from .utils import (
    allow_autoreload,
    cast_unchecked,
    fval,
    jit,
    pformat_repr,
    tree_at_,
    tree_map,
)

# pint_registry().define("eps_disp=[eps_disp]")


@jax.custom_jvp
def stop_gradient_once(x: Array) -> Array:
    assert False


@stop_gradient_once.defjvp
def _(primals: tuple[Array], tangents: tuple[Array]):
    (x,) = primals
    (dx,) = tangents
    return x, zeros_like_same_unit(dx)


def _aggregate_forces(g: graph_t) -> tuple[batched[force_annotation], batched[Array]]:

    def inner(c: connection):
        a = g.get_point(c.a)
        b = g.get_point(c.b)

        v = b.coords - a.coords
        v_len = jnp.linalg.norm(v)
        v_dir = v / v_len  # vector a->b

        # deform = v_len / lax.stop_gradient(v_len) - 1

        # > 0 : compression
        force: Array = -primal_to_zero(v_len) / v_len * c.force_per_deform

        weight = c.weight + c.density * stop_gradient_once(v_len)

        half_w = jnp.array([0, 0, weight]) / 2

        ans1 = batched.create(force_annotation(c.a, -force * v_dir - half_w))
        ans2 = batched.create(force_annotation(c.b, force * v_dir - half_w))

        return batched.stack([ans1, ans2]), force

    annos, connection_forces = g._connections.map(inner).split_tuple()

    return annos.unflatten().reshape(-1), connection_forces


def forces_from_tangents(
    g: graph_t, displacements: batched[Array]
) -> tuple[batched[Array], batched[Array]]:

    assert displacements.batch_dims() == g._points.batch_dims()
    assert displacements.item_shape().shape == (3,)

    def forces_from_coords(
        coords: batched[Array],
    ) -> tuple[batched[Array], batched[Array]]:
        g2 = tree_at_(lambda g: g._points.unflatten().coords, g, coords.unflatten())

        annos, connection_fs = _aggregate_forces(g2)

        g2 = g2.add_external_force_batched(annos)

        aggr = g2.sum_annotations()

        return aggr, connection_fs

    primals = g._points.map(lambda x: x.coords)
    (primals_out, _), (tangents_out, connection_fs) = jax.jvp(
        forces_from_coords, (primals,), (displacements,)
    )

    return (
        batched_vmap(lambda x, y: x + y, primals_out, tangents_out),
        connection_fs,
    )


class solve_forces_r[T](eqx.Module):
    graph_args: T
    graph: graph_t

    point_displacements: batched[Array]
    connection_forces: batched[Array]

    point_force_errors: batched[Array]

    __repr__ = pformat_repr


def solve_forces[T](
    graph_fn: Callable[[T], graph_t], graph_ex: graph_t, arg_ex: T
) -> solve_forces_r[T]:

    def get_equations(solve_vars: tuple[batched[Array], T]):
        tangents, arg = solve_vars
        graph = graph_fn(arg)
        tangents = tangents.map(lambda x: x * areg.m)
        equations, connection_forces = forces_from_tangents(graph, tangents)
        return equations, graph, connection_forces

    f_args = (
        graph_ex._points.map(lambda x: jnp.zeros_like(x.coords) * areg.dimensionless),
        arg_ex,
    )

    ans = flstsq(lambda x: get_equations(x)[0], f_args)

    debug_print("solve_forces: residuals=", ans.residuals)

    _, graph, connection_forces = get_equations(ans.x)

    point_displacements, graph_args = ans.x

    return solve_forces_r(
        graph_args=graph_args,
        graph=graph,
        point_displacements=point_displacements,
        connection_forces=connection_forces,
        point_force_errors=ans.errors,
    )
