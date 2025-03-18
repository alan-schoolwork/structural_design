import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax
from jax.experimental.checkify import check
from jaxtyping import Array, Float, jaxtyped

from .utils import fval, ival


@jaxtyped(typechecker=typechecker)
def _partial_sum(x: Float[Array, "p"]) -> Float[Array, "p"]:
    _, ans = lax.scan(
        lambda prev_sum, cur: (prev_sum + cur, prev_sum),
        init=0.0,
        xs=x,
    )
    assert isinstance(ans, Array)
    return ans


class force_profile(eqx.Module):
    start: fval
    stop: fval
    interval: fval
    positions: Array
    forces: Array

    @jaxtyped(typechecker=typechecker)
    def _rot_mom(self) -> Float[Array, "p"]:
        return jax.vmap(lambda f, p: f * p)(self.forces, self.positions)

    @jaxtyped(typechecker=typechecker)
    def net_force(self) -> fval:
        return jnp.sum(self.forces)

    @jaxtyped(typechecker=typechecker)
    def net_rot(self) -> fval:
        return jnp.sum(self._rot_mom())

    @jaxtyped(typechecker=typechecker)
    def shear(self) -> Float[Array, "p"]:
        return _partial_sum(self.forces)

    @jaxtyped(typechecker=typechecker)
    def moment(self) -> Float[Array, "p"]:
        # moment(x): sum((pos-x)*f for x, f in ... if pos < x)
        return _partial_sum(_partial_sum(self.forces) * self.interval)


class force_profile_builder:
    start: fval
    stop: fval
    interval: fval
    positions: Array
    forces: Array

    @jaxtyped(typechecker=typechecker)
    def __init__(self, start: fval, stop: fval, n: int):
        self.start = start
        self.stop = stop
        self.interval = (stop - start) / (n - 1)
        self.positions = jnp.linspace(start, stop, n)
        self.forces = jnp.zeros_like(self.positions)

    @jaxtyped(typechecker=typechecker)
    def assert_bounds(self, pos: fval):
        check(self.start <= pos, "assert_bounds")
        check(pos <= self.stop, "assert_bounds")

    @jaxtyped(typechecker=typechecker)
    def add_uniform(self, start: fval, stop: fval, force_density: fval):
        self.assert_bounds(start)
        self.assert_bounds(stop)

        @jaxtyped(typechecker=typechecker)
        def inner(old_f: fval, pos: fval) -> fval:
            return lax.select(
                (start <= pos) & (pos <= stop),
                old_f + force_density * self.interval,
                old_f,
            )

        ans: Array = jax.vmap(inner)(self.forces, self.positions)
        self.forces = ans

    @jaxtyped(typechecker=typechecker)
    def add_point(self, p: fval, force: fval):
        self.assert_bounds(p)
        pos_idx: ival = jnp.argmin(jnp.abs(self.positions - p))
        self.forces = self.forces.at[pos_idx].add(force)

    def build(self) -> force_profile:
        return force_profile(
            start=self.start,
            stop=self.stop,
            interval=self.interval,
            positions=self.positions,
            forces=self.forces,
        )
