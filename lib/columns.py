import math

import jax
import jax.numpy as jnp

from .utils import (
    flike,
    fval,
)


def _calc_inertia(positions: list[int], x_len: flike) -> fval:
    center = sum(positions) / len(positions)

    def inner(pos: fval):
        offset = (pos - center) * x_len
        v1 = (offset - x_len / 2) ** 3 * x_len / 3
        v2 = (offset + x_len / 2) ** 3 * x_len / 3
        return jnp.abs(v1 - v2)

    return jnp.sum(jax.vmap(inner)(jnp.array(positions)))


def calc_inertia(shape: str, x_len: flike) -> tuple[fval, fval]:
    # (x insertia, y inertia)

    def process_line(row: int, l: str):
        for column, c in enumerate(l):
            if c == "x":
                yield row, column
            else:
                assert c == "."

    def gen():
        for row, x in enumerate(shape.splitlines()):
            if (s := x.strip()) != "":
                yield from process_line(row, s)

    positions = sorted(gen())

    x_i = _calc_inertia([x for y, x in positions], x_len)
    y_i = _calc_inertia([y for y, x in positions], x_len)

    return (x_i, y_i)


def calc_buckle(
    shape: str,
    height: flike,
    x_len: flike,
    modulus_of_elasticity: flike,
    k_factor: float = 1.0,
) -> fval:
    # returns critical load in psi

    inertia = jnp.min(jnp.array(calc_inertia(shape, x_len)))
    # print(inertia)

    return (math.pi**2 * jnp.array(modulus_of_elasticity) * inertia) / (
        # column effective length factor
        # (end condition)
        k_factor
        * height
    ) ** 2
