import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import oryx
import sympy as s
import sympy2jax
from jax import Array, lax
from jax import tree_util as jtu
from jax._src.typing import ArrayLike
from jax.experimental.checkify import check, checkify
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pintax import areg, convert_unit, quantity, sync_units, unitify, ureg
from pintax._utils import pretty_print
from pintax.functions import lstsq

from lib.batched import batched, batched_vmap, do_batch, tree_do_batch, unbatch
from lib.beam import force_profile, force_profile_builder
from lib.checkify import checkify_simple
from lib.graph import graph_t, point, pointid
from lib.jax_utils import debug_print, flatten_handler
from lib.lstsq import flstsq, flstsq_checked
from lib.plot import plot_unit
from lib.utils import blike, bval, debug_callback, fval, ival, jit, unique


@jit
def circle_intersection(p1_: Array, p2_: Array, r1_v_: Array, r2_v_: Array):
    args, u = sync_units((p1_, p2_, r1_v_, r2_v_))
    p1, p2, r1_v, r2_v = map(jnp.array, args)

    x, y = s.symbols("x y")
    x1, y1, x2, y2 = s.symbols("x1 y1 x2 y2")
    r1, r2 = s.symbols("r1, r2")

    ans = s.solve(
        [
            s.Eq((x - x1) ** 2 + (y - y1) ** 2, r1**2),
            s.Eq((x - x2) ** 2 + (y - y2) ** 2, r2**2),
        ],
        [x, y],
    )
    assert len(ans) == 2
    f = lambda x: s.simplify(s.expand(x))
    ans_ = (tuple(map(f, ans[0])), tuple(map(f, ans[1])))

    ans0, ans1 = sympy2jax.SymbolicModule(ans_, make_array=False)(
        x1=p1[0],
        y1=p1[1],
        x2=p2[0],
        y2=p2[1],
        r1=r1_v,
        r2=r2_v,
    )
    u_ = u.a
    return jnp.array(ans0) * u_, jnp.array(ans1) * u_


def _polar(ang: ArrayLike, r: ArrayLike) -> Array:
    return jnp.array([jnp.cos(ang), jnp.sin(ang)]) * r


@jit
def build_graph() -> graph_t:
    print("tracing", build_graph)

    x, y, r, h = s.symbols("x y r h")

    radius = 29.5 * areg.meters
    height = 18.0 * areg.meters

    r0 = 2.5 * areg.meters
    r1 = 5.0 * areg.meters
    r2 = 22.0 * areg.meters
    r_arc = 15.0 * areg.meters

    vert_rad = 25.5 * areg.meters
    vert_rad2 = 23.0 * areg.meters
    n_support = 36
    support_res = math.pi * 2 / n_support

    sphere_center_height = -sympy2jax.SymbolicModule(
        unique(s.solve(s.Eq(x**2 + r**2, (x + h) ** 2), x)), make_array=False
    )(
        r=radius,
        h=height,
    )
    ball_rad = height - sphere_center_height

    # x**2 + y**2 + (z-(-h))**2 == r**2

    def _project_get_h(p: Array) -> Array:
        # return jnp.array(0.0)
        assert p.shape == (2,)
        # r_min = 2.0 * areg.meters
        r_min = r0
        # r_min = 0.0
        rat = jnp.maximum((jnp.linalg.norm(p) - r_min) / (radius - r_min), 0.0)
        ans = (jnp.cosh(rat) - jnp.cosh(1)) / (jnp.cosh(0) - jnp.cosh(1))
        return ans * height

        # return jnp.sqrt(ball_rad**2 - jnp.sum(p**2)) + sphere_center_height

    def add_point(
        g: graph_t,
        p: Array,
        unbatch_dims: tuple[int, ...] = (),
        *,
        fixed: blike = False,
        accepts_force: blike = False,
    ):
        h = _project_get_h(p)
        coord = jnp.array([p[0], p[1], h])
        return g.add_point(
            coord, unbatch_dims, fixed=fixed, accepts_force=accepts_force
        )

    def batched_connect(unbatch_dims: tuple[int, ...]):
        def wrapped(a: batched[pointid], b: batched[pointid]):
            nonlocal g
            g = g.add_connection(a.unwrap(), b.unwrap(), unbatch_dims)

        return wrapped

    def connect_ring(pts: batched[pointid], extra_unbatch: tuple[int, ...] = ()):
        (length,) = pts.batch_dims()
        jax.vmap(batched_connect((*extra_unbatch, length)))(pts, pts.roll(1))

    g = graph_t.create()

    # g, center_top = add_point(g, jnp.array([0.0, 0.0]) * areg.m)

    def inner(i: Array):
        nonlocal g

        ang = support_res * i
        g, outer = add_point(
            g, _polar(ang, radius), (n_support,), fixed=True, accepts_force=True
        )

        tmp = _polar(ang, vert_rad)
        g, vert_top = add_point(g, tmp, (n_support,))
        g, vert_top_proj = g.add_point(
            jnp.array([tmp[0], tmp[1], 0.0]),
            (n_support,),
            fixed=True,
            accepts_force=True,
        )
        g = g.add_connection(vert_top, vert_top_proj, (n_support,))

        tmp = _polar(ang + support_res / 2, vert_rad2)
        g, vert_top2 = add_point(g, tmp, (n_support,))
        connect_ring(batched.create_unbatch(vert_top2, (n_support,)))
        # g, vert_top2_proj = g.add_point(
        #     jnp.array([tmp[0], tmp[1], 0.0]), (n_support,), fixed=True
        # )
        # g = g.add_connection(vert_top2, vert_top2_proj, (n_support,))

        g = g.add_connection(vert_top, outer, (n_support,))

        g = g.add_connection(vert_top, vert_top2, (n_support,))

        vert_top2_shifted = tree_do_batch(
            batched.create_unbatch(vert_top2, (n_support,)).roll(1), dims=(n_support,)
        ).unwrap()
        g = g.add_connection(vert_top, vert_top2_shifted, (n_support,))

        return batched.create(vert_top2)

    vert_top2 = jax.vmap(inner)(jnp.arange(n_support))

    def draw_circle(c: Array, r: Array):
        res = 300

        def for_ang(ang: Array):
            nonlocal g
            g, ans = add_point(g, c + _polar(ang, r), (res,))
            return batched.create(ans)

        pts = jax.vmap(for_ang)(jnp.arange(res) / res * 2 * math.pi)
        jax.vmap(batched_connect((res,)))(pts, batched.concat([pts[1:], pts[:1]]))

    # draw_circle(jnp.array([0.0, 0.0]) * areg.m, r1)
    # draw_circle(jnp.array([0.0, 0.0]) * areg.m, r2)
    # draw_circle(jnp.array([0.0, 0.0]) * areg.m, radius)

    def _arc_centers(p1: Array, p2: Array):
        m = (p1 + p2) / 2
        vc = p2 - p1
        dist = jnp.sqrt(r_arc**2 - jnp.sum((vc / 2) ** 2))
        vc_x, vc_y = vc / jnp.linalg.norm(vc) * dist
        v = jnp.array([vc_y, -vc_x])
        return m + v

    # n_arcs = 36
    n_arcs = 108
    arc_skip = n_arcs // 12
    n_rings = arc_skip * 2 + 1

    ang_res = math.pi * 2 / n_arcs

    def inner2(i: Array):
        a1 = ang_res * i
        a2 = a1 - ang_res * arc_skip
        a3 = a1 + ang_res * arc_skip

        p1 = _polar(a1, r1)
        p2 = _polar(a2, r2)
        p3 = _polar(a3, r2)

        c1 = _arc_centers(p3, p1)
        c2 = _arc_centers(p1, p2)
        return batched.create(c1), batched.create(c2)

    c1s, c2s = jax.vmap(inner2)(jnp.arange(n_arcs))

    # for i in range(10):
    #     draw_circle(c1s[i].unwrap(), r_arc)
    # draw_circle(c1s[1].unwrap(), r_arc)
    # draw_circle(c2s[0].unwrap(), r_arc)
    # draw_circle(c2s[1].unwrap(), r_arc)
    # draw_circle(c2s[arc_skip * 2].unwrap(), r_arc)

    # draw_circle(c2s[1].unwrap(), r_arc)

    # print(c1s[0].unwrap())
    # print(c2s[0].unwrap())
    # cs1
    # tmp = circle_intersection(
    #     c1s[0].unwrap(),
    #     c2s[0].unwrap(),
    #     r_arc,
    #     r_arc,
    # )
    # print("tmp", tmp)

    c2s_dup = batched.concat([c2s, c2s])

    def one_arc(idx: Array):
        intersect_with = c2s_dup.dynamic_slice((idx,), (n_rings,))

        def one_intersection(c2: batched[Array]):
            nonlocal g
            it, _ = circle_intersection(
                c1s[idx].unwrap(),
                c2.unwrap(),
                r_arc,
                r_arc,
            )
            g, itp = add_point(g, it, (n_arcs, n_rings))
            return batched.create(itp)

        ips = jax.vmap(one_intersection)(intersect_with)
        jax.vmap(batched_connect((n_arcs, n_rings - 1)))(ips[:-1], ips[1:])
        return ips

    net = jax.vmap(one_arc)(jnp.arange(n_arcs))
    assert net.batch_dims() == (n_arcs, n_rings)

    def reverse_arc(idx: Array):

        def inner(idx2: Array):
            idxw = idx - idx2
            idxw = lax.select(idxw >= 0, on_true=idxw, on_false=idxw + n_arcs)
            ans = net[idxw][idx2]
            assert ans.batch_dims() == ()
            return ans

        rev_arc = jax.vmap(inner)(jnp.arange(n_rings))

        jax.vmap(batched_connect((n_arcs, n_rings - 1)))(rev_arc[:-1], rev_arc[1:])

    jax.vmap(reverse_arc)(jnp.arange(n_arcs))

    def base_conections(idx: Array):
        vt = vert_top2[idx]

        def connect_one(idx2: Array):
            idx_ = idx * 3 - arc_skip + idx2
            idx_ = lax.select(idx_ >= 0, idx_, idx_ + n_arcs)
            idx_ = lax.select(idx_ < n_arcs, idx_, idx_ - n_arcs)

            batched_connect((n_support, 4))(vert_top2[idx], net[idx_, -1])

        jax.vmap(connect_one)(jnp.array([0, 1, 2, 3]))

    jax.vmap(base_conections)(jnp.arange(n_support))

    def connect_net_layer(l: int):
        return connect_ring(net[:, l])

    def connect_net_layer_for_all(pts: batched[pointid]):
        return connect_ring(pts, extra_unbatch=(n_rings,))

    jax.vmap(connect_net_layer_for_all, in_axes=1)(net)

    # connect_net_layer(-1)
    # # connect_net_layer(-2)
    # # connect_net_layer(-3)
    # # connect_net_layer(1)
    # connect_net_layer(0)

    def _make_inner_ring(i: Array):
        nonlocal g
        a = ang_res * i
        g, inner_ring = add_point(g, _polar(a, r0), (n_arcs,), fixed=True)
        # g = g.add_connection(inner_ring, center_top, (n_arcs,))

        return batched.create(inner_ring)

    inner_ring = jax.vmap(_make_inner_ring)(jnp.arange(n_arcs))

    connect_ring(inner_ring)
    jax.vmap(batched_connect((n_arcs,)))(inner_ring, net[:, 0])

    # n_rings
    # print("i1s", ips)

    # print("i1", i1)
    # print("i2", i2)

    # g, i1p = add_point(g, i1)
    # g, i2p = add_point(g, i2)
    # g = g.add_connection(i1p, i2p)

    # p1 =

    # tmp = jnp.array([0.0, 60.0]) * areg.m

    return g
