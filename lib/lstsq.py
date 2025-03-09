from dataclasses import dataclass
from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype import beartype as typechecker
from jax import tree_util as jtu
from jax.experimental.checkify import check
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped
from pintax.functions import lstsq

from .utils import bval, fval, ival


class flstsq_r[T, R](eqx.Module):
    const: R

    x: T
    residuals: Array
    rank: Array
    svals: Array

    # other outputs here if needed


def flstsq[T, R](f: Callable[[T], R], arg_example: T) -> flstsq_r[T, R]:
    """
    A functional version of lstsq, returning both the
    updated input x and const as the function output at
    the original point.
    """
    # Flatten example input into a single array.
    arg_bufs, arg_tree = jtu.tree_flatten(arg_example)
    for x in arg_bufs:
        assert x.dtype == arg_bufs[0].dtype, "All inputs must have the same dtype."

    def flatten_args(args_list):
        return jnp.concatenate([a.ravel() for a in args_list], axis=0)

    def split_args(args_flat):
        """Invert flatten_args: split args_flat into shapes like arg_bufs."""
        offset = 0
        pieces = []
        for buf in arg_bufs:
            size = buf.size
            piece = args_flat[offset : offset + size].reshape(buf.shape)
            pieces.append(piece)
            offset += size
        return pieces

    # Flatten the original argument into one big 1D array.
    args_concatenated = flatten_args(arg_bufs)

    # Example function output structure, to help shape the flattened outputs.
    out_example = f(arg_example)
    out_bufs_example, out_tree_example = jtu.tree_flatten(out_example)

    def flatten_out(out_val):
        """Flatten the function output to a 1D array."""
        bufs, _ = jtu.tree_flatten(out_val)
        return jnp.concatenate([b.ravel() for b in bufs], axis=0)

    def split_out(out_flat):
        """Unflatten a 1D output array back into the original output structure."""
        offset = 0
        pieces = []
        for b in out_bufs_example:
            size = b.size
            piece = out_flat[offset : offset + size].reshape(b.shape)
            pieces.append(piece)
            offset += size
        return jtu.tree_unflatten(out_tree_example, pieces)

    def inner(args_flat: jnp.ndarray):
        # From the single 1D array, reconstruct inputs as a PyTree.
        ipt = jtu.tree_unflatten(arg_tree, split_args(args_flat))
        # Evaluate the user function.
        out_val = f(ipt)
        out_flat = flatten_out(out_val)
        # Return (output, output) so the second can be treated as aux by jax.jacobian.
        return out_flat, out_flat

    # Compute the Jacobian (mat) of inner w.r.t. inputs, and also get “const” as the aux part.
    mat, const_flat = jax.jacobian(inner, has_aux=True)(args_concatenated)
    # Now mat has shape [output_dim, input_dim], and const_flat has shape [output_dim].

    # Solve the linear system mat @ δx ≈ −const (a least-squares problem).
    delta_args_flat, residuals, rank, svals = lstsq(mat, -const_flat, rcond=None)

    # Construct the new input by adding δx to the original arg_bufs.
    updated_bufs = [
        orig + δ.reshape(orig.shape)  # reshape in case the dimension matches
        for orig, δ in zip(arg_bufs, split_args(delta_args_flat))
    ]
    new_x = jtu.tree_unflatten(arg_tree, updated_bufs)

    # Unflatten const_flat so it matches the structure of the original function output.
    const_unflat = split_out(const_flat)

    # Return the result dataclass with the original output as const and updated input x.
    return flstsq_r(
        const=const_unflat, x=new_x, residuals=residuals, rank=rank, svals=svals
    )


def flstsq_checked[T, R](f: Callable[[T], R], arg_example: T) -> flstsq_r[T, R]:
    ans = flstsq(f, arg_example)
    check(jnp.all(ans.residuals < 10 ** (-5)), "flstsq_checked")
    return ans
