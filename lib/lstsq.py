from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax import tree_util as jtu
from jax.experimental.checkify import check
from jaxtyping import Array

from pintax.functions import lstsq

from .jax_utils import debug_print, flatten_handler, fn_as_traced
from .utils import cast_unchecked, pformat_repr


class flstsq_r[T, R](eqx.Module):
    const: R
    mat_flat: Array

    x_flat: Array
    x: T
    errors: R
    total_error: Array
    linear_residuals: Array
    rank: Array
    singular_values: Array

    __repr__ = pformat_repr


def _jac(fn: Callable[[Array], Array], at: Array) -> tuple[Array, Array]:
    def inner(x):
        ans = fn(x)
        return ans, ans

    mat, const = jax.jacobian(inner, has_aux=True)(at)
    return mat, const


def flstsq[T, R](f: Callable[[T], R], arg_start: T, n_iters: int = 1) -> flstsq_r[T, R]:
    """
    A functional version of lstsq, returning both the
    updated input x and const as the function output at
    the original point.
    """

    arg_tree, arg_start_flat = flatten_handler.create(arg_start)
    ans_tree = cast_unchecked[flatten_handler[R]]()(None)

    def f_flat(arg_flat: Array) -> Array:
        nonlocal ans_tree

        arg = arg_tree.unflatten(arg_flat)
        ans = f(arg)
        ans_tree, ans_flat = flatten_handler.create(ans)
        return ans_flat

    f_flat = fn_as_traced(f_flat)(arg_start_flat)

    def solve_once(cur_arg: Array):

        mat_from_cur, const_from_cur = _jac(f_flat, cur_arg)

        def linearized_fn(new_arg: Array) -> Array:
            return mat_from_cur @ (new_arg - cur_arg) + const_from_cur

        mat, const = _jac(linearized_fn, arg_start_flat)

        delta_arg, resid, rank, singular_values = lstsq(
            mat,
            -const,
            # rcond=0.01,
            rcond=None,
        )

        ans = arg_start_flat + delta_arg
        errors = f_flat(ans)
        total_error = jnp.sum(jnp.square(errors))

        debug_print(
            "flstsq:",
            "linear_residuals=",
            resid,
            "rank=",
            rank,
            "total_error=",
            total_error,
            # "mid_z_offset_a=",
            # arg_tree.unflatten(ans)[1]["mid_z_offset_a"],
        )

        return flstsq_r(
            mat_flat=mat,
            const=ans_tree.unflatten(const),
            #
            x_flat=ans,
            x=arg_tree.unflatten(ans),
            errors=ans_tree.unflatten(errors),
            total_error=total_error,
            linear_residuals=resid,
            rank=rank,
            singular_values=singular_values,
        )

    solve_once = fn_as_traced(solve_once)(arg_start_flat)

    ans = solve_once(arg_start_flat)

    if n_iters > 1:
        _, ans = lax.while_loop(
            lambda i_s: (i_s[0] < n_iters - 1) & (i_s[1].total_error != 0.0),
            lambda i_s: (i_s[0] + 1, solve_once(i_s[1].x_flat)),
            (jnp.array(0), ans),
        )

    return ans


def flstsq_checked[T, R](f: Callable[[T], R], arg_example: T) -> flstsq_r[T, R]:
    ans = flstsq(f, arg_example)
    check(jnp.all(ans.linear_residuals < 10 ** (-5)), "flstsq_checked")
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
