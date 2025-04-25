# pyright: reportUnusedVariable=false
import math
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import oryx
from jax import Array, lax
from jax import tree_util as jtu
from jax.typing import ArrayLike
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pintax.oryx_ext as _
from final.graph import build_graph, tag_external_force
from lib.batched import batched, batched_vmap, batched_zip
from lib.graph import connection, force_annotation, graph_t, point
from lib.jax_utils import fn_as_traced, oryx_unzip
from lib.lstsq import flstsq
from lib.plot import plot_graph_args, plot_graph_forces
from lib.utils import allow_autoreload, concatenate, fval, jit, tree_at_, vmap
from pintax import areg, unitify, ureg

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

np.set_printoptions(precision=3, suppress=True)

# matplotlib.use("tkagg")


@allow_autoreload
@partial(unitify, unwrap_outs=True)
def main():

    graph_fn = fn_as_traced(build_graph)()
    # graph_fn = build_graph

    graph_ex = graph_fn()

    extra_vars, graph_fn = oryx_unzip(graph_fn, tag=tag_external_force)

    from lib.displacement_based_forces import solve_forces

    # from lib.simple_solve import solve_forces

    ans = solve_forces(graph_fn, graph_ex, extra_vars)

    # ans = jit(solve_forces)(graph_fn, graph_ex, extra_vars)

    plot_graph_forces(
        plot_graph_args(
            graph=ans.graph,
            connection_forces=ans.connection_forces,
            f_max=500.0 * areg.weight_c,
        )
    )
    # plt.show()
    plt.tight_layout()
    plt.savefig("output.png", dpi=300)
    plt.close()

    return ans
