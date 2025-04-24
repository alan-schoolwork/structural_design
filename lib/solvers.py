import abc
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax import tree_util as jtu
from jax.experimental.checkify import check
from jaxtyping import Array
from optax import OptState

from pintax import quantity
from pintax.functions import lstsq

from .jax_utils import debug_print, flatten_handler, fn_as_traced, jaxpr_with_tree
from .utils import cast, cast_unchecked, cond_, jit, pformat_repr, value_and_grad_aux_


class SolverImplRes[E](eqx.Module):
    arg_out: Array

    extras: E
    arglike_annotations: dict[str, Array] = eqx.field(default_factory=lambda: {})
    reslike_annotations: dict[str, Array] = eqx.field(default_factory=lambda: {})


class SolverRes[T, R, E](eqx.Module):
    arg_out: T
    res_out: R

    extras: E
    arglike_annotations: dict[str, T]
    reslike_annotations: dict[str, R]


class Solver[E](abc.ABC):

    @abstractmethod
    def solve_impl(
        self, f: jaxpr_with_tree[[Array], Array], init: Array
    ) -> SolverImplRes[E]: ...

    def solve[T, R](self, f: Callable[[T], R], init: T) -> SolverRes[T, R, E]:
        arg_tree, init_flat = flatten_handler.create(init)
        ans_tree = cast[flatten_handler[R] | None]()(None)

        def f_flat(arg_flat: Array) -> Array:
            nonlocal ans_tree

            arg = arg_tree.unflatten(arg_flat)
            ans = f(arg)
            ans_tree, ans_flat = flatten_handler.create(ans)
            return quantity(ans_flat).m_arr

        f_flat = fn_as_traced(f_flat)(init_flat)
        assert ans_tree is not None
        ans = self.solve_impl(f_flat, init_flat)

        return SolverRes(
            arg_out=arg_tree.unflatten(ans.arg_out),
            res_out=ans_tree.unflatten(f_flat(ans.arg_out)),
            #
            extras=ans.extras,
            arglike_annotations={
                x: arg_tree.unflatten(y) for x, y in ans.arglike_annotations.items()
            },
            reslike_annotations={
                x: ans_tree.unflatten(y) for x, y in ans.reslike_annotations.items()
            },
        )


class GradientDescentRes(eqx.Module):
    pass


_gd_state_t = tuple[Array, optax.OptState]


@dataclass
class GradientDescent(Solver[GradientDescentRes]):
    learning_rate: float = 10 ** (-4)
    n_steps: int = 10

    def solve_impl(
        self, f: jaxpr_with_tree[[Array], Array], init: Array
    ) -> SolverImplRes[GradientDescentRes]:

        optimizer = optax.adam(learning_rate=self.learning_rate)
        init_opt_state: OptState = optimizer.init(init)

        def loss_fn(x: Array):
            ans = f(x)
            return jnp.sum(jnp.square(ans))

        def optim_loop(state: _gd_state_t, idx: Array) -> tuple[_gd_state_t, None]:
            x, opt_state = state
            loss, grads = jax.value_and_grad(loss_fn)(x)

            cond_(
                idx % 10000 == 0,
                true_fun=lambda: debug_print(
                    "step:",
                    idx,
                    "loss:",
                    loss,
                    # jnp.max(grads[0]),
                    # jnp.min(grads[0]),
                ),
                false_fun=lambda: None,
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            buffers = optax.apply_updates(x, updates)
            assert isinstance(buffers, Array)
            return (buffers, opt_state), None

        (ans, _opt_state), _ = jit(
            lambda: lax.scan(
                optim_loop,
                init=(init, init_opt_state),
                length=self.n_steps,
                xs=jnp.arange(self.n_steps),
            )
        )()

        return SolverImplRes(ans, GradientDescentRes())


class LstsqRes(eqx.Module):
    total_error: Array
    # const: R
    # mat_flat: Array

    # x_flat: Array
    # x: T
    # errors: R
    # total_error: Array
    # linear_residuals: Array
    # rank: Array
    # singular_values: Array

    # __repr__ = pformat_repr


def _jac(fn: Callable[[Array], Array], at: Array) -> tuple[Array, Array]:
    def inner(x):
        ans = fn(x)
        return ans, ans

    mat, const = jax.jacobian(inner, has_aux=True)(at)
    return mat, const


@dataclass
class Lstsq(Solver[LstsqRes]):
    n_iters: int = 1

    def solve_impl(
        self, f: jaxpr_with_tree[[Array], Array], init: Array
    ) -> SolverImplRes[LstsqRes]:
        def solve_once(cur_arg: Array):

            mat_from_cur, const_from_cur = _jac(f, cur_arg)

            def linearized_fn(new_arg: Array) -> Array:
                return mat_from_cur @ (new_arg - cur_arg) + const_from_cur

            mat, const = _jac(linearized_fn, init)

            delta_arg, resid, rank, singular_values = lstsq(
                mat,
                -const,
                # rcond=1e-5,
                rcond=None,
            )

            ans = init + delta_arg
            errors = f(ans)
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

            return SolverImplRes(
                ans,
                LstsqRes(
                    total_error=total_error,
                ),
            )

        solve_once = fn_as_traced(solve_once)(init)

        ans = solve_once(init)

        if self.n_iters > 1:
            _, ans = lax.while_loop(
                lambda i_s: (
                    (i_s[0] < self.n_iters - 1) & (i_s[1].extras.total_error > 1e-8)
                ),
                lambda i_s: (i_s[0] + 1, solve_once(i_s[1].arg_out)),
                (jnp.array(0), ans),
            )

        return ans
