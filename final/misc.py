# pyright: reportUnusedCallResult=false
import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import oryx
from beartype import beartype as typechecker
from jax import Array, lax
from jax import tree_util as jtu
from jax.typing import ArrayLike
from jaxtyping import Float, jaxtyped
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pintax.oryx_ext as _
from lib.batched import batched, batched_vmap, batched_zip
from lib.displacement_based_forces import solve_forces
from lib.graph import connection, force_annotation, graph_t, point
from lib.jax_utils import fn_as_traced, oryx_unzip
from lib.lstsq import flstsq
from lib.utils import allow_autoreload, concatenate, fval, jit, tree_at_, vmap
from pintax import areg, quantity, unitify, ureg


@allow_autoreload
@unitify
def misc():
    ang = 360 / 12 / 2 * areg.degrees

    print("compression ring", 250 * areg.kpounds / jnp.sin(ang) / 2)


@allow_autoreload
@unitify
def misc2():

    outer_r = 178.712 * areg.ft / 2
    inner_r = 35.682 * areg.ft / 2

    print("len top/bot", (outer_r - inner_r) * 12)

    print("outer", outer_r * 2 * math.pi)
    print("inner", inner_r * 2 * math.pi)


def force_diagram():
    outer_r = 178.712 / 2
    inner_r = 35.682 / 2

    n = 12 + 1
    x_pos = jnp.linspace(outer_r, inner_r, n)

    forces = x_pos / x_pos.sum() * 167

    forces_s: list[float] = [0.0] + forces.cumsum().tolist()

    ans1: list[tuple[tuple[float, float], tuple[float, float]]] = []
    ans2: list[tuple[tuple[float, float], tuple[float, float]]] = []

    points_star = [(0, x) for x in forces_s]

    tmp = math.sqrt((outer_r - inner_r) ** 2 + 20**2)
    f_top = 216.89

    points = [
        (-f_top * (outer_r - inner_r) / tmp, -f_top * 20 / tmp + x) for x in forces_s
    ][1:-1]

    ans1 += list(zip(points[1:], points[:-1]))
    ans1 += list(zip(points_star[1:], points_star[:-1]))
    ans1 += list(zip(points, points_star[1:-1]))

    y = points_star[0]
    x = points_star[-1]

    o = (221.112, x[1])
    z = (221.112, y[1])

    ans2 += [(p, o) for p in points]
    ans2 += [(x, o), (z, y)]
    ans1 += [(o, z)]

    for p in points:
        print(math.sqrt((p[0] - o[0]) ** 2 + (p[1] - o[1]) ** 2))

    fig, ax = plt.subplots()

    lc = LineCollection(ans1)
    lc.set_alpha(0.5)
    lc.set_color("red")
    ax.add_collection(lc)

    lc = LineCollection(ans2)
    lc.set_alpha(0.5)
    lc.set_color("blue")
    ax.add_collection(lc)

    ax.set_xlabel("force (kip)")
    ax.set_ylabel("force (kip)")

    ax.autoscale()
    ax.set_aspect("equal")

    # plt.show()

    fig.tight_layout()
    fig.savefig("force_diagram.png", dpi=300)
    plt.close(fig)

    # print(forces_s)

    pass
