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
from jax import tree_util as jtu
from jax.experimental.checkify import check, checkify
from pintax import areg, convert_unit, quantity, unitify, ureg
from pintax._utils import pretty_print
from pintax.functions import lstsq

from lib.beam import force_profile, force_profile_builder
from lib.checkify import checkify_simple
from lib.graph import graph, graph_builder
from lib.jax_utils import debug_print, flatten_handler
from lib.lstsq import flstsq, flstsq_checked
from lib.plot import plot_unit
from lib.utils import bval, debug_callback, fval, ival, jit, unique

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", True)

np.set_printoptions(precision=3, suppress=True)


@unitify
def main():

    x, r, h = s.symbols("x r h")

    def build_graph() -> graph:

        print("tracing", build_graph)

        g = graph_builder.create(
            default_dist=areg.inch,
            default_force=areg.force_pound,
        )

        radius = 29.5 * areg.meters
        height = 18.0 * areg.meters
        n_support = 36

        sphere_center_height = -sympy2jax.SymbolicModule(
            unique(s.solve(s.Eq(x**2 + r**2, (x + h) ** 2), x)), make_array=False
        )(
            r=radius,
            h=height,
        )
        print(sphere_center_height)

        a1 = g.point(0.0, 0.0)
        assert False

    build_graph()
