from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import oryx
from beartype import beartype as typechecker
from jax import tree_util as jtu
from jax._src.typing import ArrayLike, DType, Shape
from jax.experimental.checkify import check, checkify
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped
from matplotlib.axes import Axes
from pintax import areg, quantity, unitify, ureg
from pintax.functions import lstsq

from lib.beam import force_profile, force_profile_builder
from lib.jax_utils import flatten_handler
from lib.lstsq import flstsq, flstsq_checked
from lib.utils import bval, fval, ival, jit


@jaxtyped(typechecker=typechecker)
def plot_unit(
    x: Float[Array, "p"],
    y: Float[Array, "p"],
    ax: Axes | None = None,
):
    if ax is None:
        ax = plt.gca()

    x1 = quantity(x)
    y1 = quantity(y)

    y_min_v = float(jnp.min(y) / y1.u)
    y_max_v = float(jnp.max(y) / y1.u)

    x1_u = x1.u._pretty_print().format(use_color=False)
    y1_u = y1.u._pretty_print().format(use_color=False)

    ax.axhline(y=0, color="gray", linestyle="-")
    ax.axhline(y=y_min_v, color="red", linestyle=":")
    ax.axhline(y=y_max_v, color="red", linestyle=":")
    ax.plot(np.array(x1.m), np.array(y1.m), color="blue")
    ax.set_xlabel(x1_u)
    ax.set_ylabel(f"{y1_u} min={y_min_v:.1f} max={y_max_v:.1f}")

    # if file is not None:
    #     ax.subplots_adjust(left=0.2)
    #     ax.savefig(file, dpi=300)
    #     ax.clf()
