import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from matplotlib.axes import Axes
from pintax import quantity


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
