from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import tree_util as jtu
from jax.experimental.checkify import check
from jaxtyping import Array

from lib.utils import pformat_repr
from pintax.functions import lstsq


class flstsq_r[T, R](eqx.Module):
    const: R

    x: T
    errors: R
    residuals: Array

    __repr__ = pformat_repr
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
    # delta_args_flat, _, _, _ = lstsq(mat, -const_flat, rcond=None)
    delta_args_flat = lstsq_safe(mat, -const_flat)
    # assert False

    errors_flat = mat @ delta_args_flat + const_flat

    # Construct the new input by adding δx to the original arg_bufs.
    updated_bufs = [
        orig + δ.reshape(orig.shape)  # reshape in case the dimension matches
        for orig, δ in zip(arg_bufs, split_args(delta_args_flat))
    ]
    new_x = jtu.tree_unflatten(arg_tree, updated_bufs)

    # Unflatten const_flat so it matches the structure of the original function output.
    const_unflat = split_out(const_flat)
    # mat @

    # Return the result dataclass with the original output as const and updated input x.
    return flstsq_r(
        const=const_unflat,
        x=new_x,
        errors=split_out(errors_flat),
        residuals=jnp.sum(errors_flat**2),
    )


def flstsq_checked[T, R](f: Callable[[T], R], arg_example: T) -> flstsq_r[T, R]:
    ans = flstsq(f, arg_example)
    check(jnp.all(ans.residuals < 10 ** (-5)), "flstsq_checked")
    return ans


@jax.custom_jvp
def lstsq_safe(a_mat: Array, b_vect: Array) -> Array:
    orig_pinv = jnp.linalg.pinv(a_mat, rtol=0.0001)
    return orig_pinv @ b_vect
    # ans, _, _, _ = lstsq(a_mat, b_vect)
    # return ans


@lstsq_safe.defjvp
def _(primals: tuple[Array, Array], tangents: tuple[Array, Array]):

    # https://github.com/jax-ml/jax/issues/10805
    a_mat, b_vect = primals
    assert len(a_mat.shape) == 2
    assert len(b_vect.shape) == 1
    assert a_mat.shape[0] == b_vect.shape[0]

    orig_pinv = jnp.linalg.pinv(a_mat, rtol=0.0001)

    primal_ans, _, _, _ = lstsq(a_mat, b_vect)
    primal_ans = orig_pinv @ b_vect

    deriv_of_a, deriv_of_b = tangents
    true_deriv_shift_b = orig_pinv @ deriv_of_b

    shift_a_deriv_pinv = (
        -orig_pinv @ deriv_of_a @ orig_pinv
        + (
            orig_pinv
            @ orig_pinv.T
            @ deriv_of_a.T
            @ (jnp.eye(b_vect.size) - a_mat @ orig_pinv)
        )
        + (
            (jnp.eye(a_mat.shape[1]) - orig_pinv @ a_mat)
            @ deriv_of_a.T
            @ orig_pinv.T
            @ orig_pinv
        )
    )
    true_deriv_shift_a = shift_a_deriv_pinv @ b_vect

    return primal_ans, true_deriv_shift_a + true_deriv_shift_b
