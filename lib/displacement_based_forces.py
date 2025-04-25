from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, lax
from matplotlib.collections import LineCollection

from lib.solvers import GradientDescent, Lstsq, SolverRes
from pintax import areg, quantity, unitify
from pintax._helpers import zeros_like_same_unit
from pintax.unstable import pint_registry

from .batched import batched, batched_vmap, batched_zip
from .graph import connection, force_annotation, graph_t, point
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
    tree_select,
)

pint_registry().define("eps_disp=[eps_disp]")


def _forces_by_disps_one(
    c: connection, a: point, b: point, da: Array, db: Array
) -> Array:
    def get_length(a_coord: Array, b_coord: Array):
        return jnp.linalg.norm(a_coord - b_coord)

    normal_val: Array
    normal_val, d_length = jax.jvp(get_length, (a.coords, b.coords), (da, db))
    deformation: Array = d_length / normal_val

    # > 0 : compression
    return tree_select(
        (c.tension_only & (deformation < 0.0))
        | (c.compression_only & (deformation > 0.0)),
        on_true=jnp.array(0.0),
        on_false=-deformation * c.force_per_deform,
    )


def forces_by_disps(g: graph_t, displacements: batched[Array]) -> batched[Array]:
    def inner(c: connection) -> Array:
        a = g.get_point(c.a)
        b = g.get_point(c.b)

        da = displacements[c.a._idx].unwrap()
        db = displacements[c.b._idx].unwrap()

        return _forces_by_disps_one(c, a, b, da, db)

    return g._connections.map(inner)


def aggregate_all_forces(
    g: graph_t, displacements: batched[Array]
) -> tuple[batched[Array], batched[Array]]:

    by_disp = forces_by_disps(g, displacements)

    def inner(c: connection, disp_force: Array):
        a = g.get_point(c.a)
        b = g.get_point(c.b)

        v = b.coords - a.coords
        v_len = jnp.linalg.norm(v)
        v_dir = v / v_len  # vector a->b

        weight = c.weight + c.density * v_len
        half_w = jnp.array([0, 0, weight]) / 2

        _scale = quantity(weight).u / quantity(disp_force).u
        debug_print("scaling disp_force:", _scale.a)

        disp_force = disp_force * _scale.a

        ans1 = batched.create(force_annotation(c.a, -disp_force * v_dir - half_w))
        ans2 = batched.create(force_annotation(c.b, disp_force * v_dir - half_w))

        return batched.stack([ans1, ans2]), disp_force

    annos, by_disp = batched_zip(g._connections, by_disp).tuple_map(inner).split_tuple()
    g = g.add_external_force_batched(annos.uf)
    aggr = g.sum_annotations()

    return aggr, by_disp


class solve_forces_r[T](eqx.Module):
    solver_out: SolverRes[tuple[batched[Array], T], batched[Array], Any]

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
        disps, arg = solve_vars
        graph = graph_fn(arg)
        equations, connection_forces = aggregate_all_forces(
            graph, disps.map(lambda x: x * areg.eps_disp)
        )
        return equations, graph, connection_forces

    f_args = (
        graph_ex._points.map(lambda x: jnp.zeros_like(x.coords) * areg.dimensionless),
        arg_ex,
    )

    _solver = GradientDescent(n_steps=100000, learning_rate=1.0)
    solver = Lstsq(n_iters=10)

    ans = solver.solve(lambda x: get_equations(x)[0], f_args)

    # debug_print("solve_forces: residuals=", ans.residuals)

    _, graph, connection_forces = get_equations(ans.arg_out)

    point_displacements, graph_args = ans.arg_out

    return solve_forces_r(
        solver_out=ans,
        graph_args=graph_args,
        graph=graph,
        point_displacements=point_displacements,
        connection_forces=connection_forces,
        point_force_errors=ans.res_out,
    )
