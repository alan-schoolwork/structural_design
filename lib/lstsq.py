from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.experimental.checkify import check
from jaxtyping import Array

from lib.jax_utils import flatten_handler
from lib.utils import cast_unchecked, pformat_repr
from pintax.functions import lstsq


class flstsq_r[T, R](eqx.Module):
    const: R
    mat_flat: Array

    x: T
    errors: R
    residuals: Array
    rank: Array
    singular_values: Array

    __repr__ = pformat_repr


def flstsq[T, R](f: Callable[[T], R], arg_example: T) -> flstsq_r[T, R]:
    """
    A functional version of lstsq, returning both the
    updated input x and const as the function output at
    the original point.
    """

    arg_tree, arg_flat = flatten_handler.create(arg_example)
    ans_tree = cast_unchecked[flatten_handler[R]]()(None)

    def inner(arg_flat: Array) -> tuple[Array, Array]:
        nonlocal ans_tree

        arg = arg_tree.unflatten(arg_flat)
        ans = f(arg)
        ans_tree, ans_flat = flatten_handler.create(ans)
        return ans_flat, ans_flat

    mat, const_flat = jax.jacobian(inner, has_aux=True)(arg_flat)

    assert isinstance(mat, Array)
    assert isinstance(const_flat, Array)

    delta_arg_flat, resid, rank, singular_values = lstsq(mat, -const_flat, rcond=None)

    errors_flat = mat @ delta_arg_flat + const_flat

    return flstsq_r(
        mat_flat=mat,
        const=ans_tree.unflatten(const_flat),
        #
        x=arg_tree.unflatten(arg_flat + delta_arg_flat),
        errors=ans_tree.unflatten(errors_flat),
        residuals=resid,
        rank=rank,
        singular_values=singular_values,
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
