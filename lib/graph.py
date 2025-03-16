from __future__ import annotations

import itertools
from dataclasses import dataclass

import equinox as eqx
import sympy as s
from jax import Array, lax
from jax import numpy as jnp
from jax.typing import ArrayLike
from pintax import areg, convert_unit

from .utils import cast_unchecked


@dataclass
class point_builder:
    name: str
    coords: Array

    def __repr__(self):
        return f"point({self.name})"

    def __init__(self, name: str, coords: Array):
        self.name = name
        self.coords = coords

    def __eq__(self, other: point_builder):
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def check_unit(self, unit: Array):
        convert_unit(self.to_jax(), unit)

    def to_jax(self):
        return jnp.array(self.coords)

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]


@dataclass
class connection_builder:
    name: str
    a: point_builder
    b: point_builder
    area: Array | None

    # >0 ==> compression
    force: Array


@dataclass
class graph_builder:
    _points: list[point_builder]
    _connections: list[connection_builder]
    _external_forces: list[tuple[point_builder, Array]]

    _pointname_counter: itertools.count[int]

    _default_dist: Array
    _default_force: Array

    @staticmethod
    def create(
        default_dist: Array,
        default_force: Array,
    ) -> graph_builder:
        # check_unit(default_dist, ctx.inch)
        # check_unit(default_force, ctx.pound)

        return graph_builder(
            _points=[],
            _connections=[],
            _external_forces=[],
            _pointname_counter=itertools.count(),
            _default_dist=default_dist,
            _default_force=default_force,
        )

    def build(self) -> graph:
        points = {p.name: p.to_jax() for p in self._points}
        forces: dict[str, Array] = {
            x: jnp.zeros_like(y) * self._default_force for x, y in points.items()
        }
        for x, y in self._external_forces:
            forces[x.name] += y

        return graph(
            points=points,
            connections={
                c.name: connection(
                    a=c.a.name,
                    b=c.b.name,
                    area=jnp.array(c.area),
                    force=jnp.array(c.force),
                )
                for c in self._connections
            },
            external_forces=forces,
        )

    def _ck_point(self, a: point_builder):
        found = False
        for x in self._points:
            if x.name == a.name:
                assert x is a
                found = True
        if not found:
            self._points.append(a)

    def point(self, name: str | ArrayLike, *coords: ArrayLike) -> point_builder:
        if not isinstance(name, str):
            coords = cast_unchecked(coords)((name, *coords))
            name = f"P{next(self._pointname_counter)}"
        else:
            name = f"{name}{next(self._pointname_counter)}"

        p = point_builder(name, jnp.array(coords))
        self._ck_point(p)
        return p

    def connection(
        self,
        a: point_builder,
        b: point_builder,
        *,
        area: ArrayLike = 0.0,
        force: ArrayLike = 0.0,
    ):
        self._ck_point(a)
        self._ck_point(b)

        assert a != b
        for x in self._connections:
            assert not (a == x.a and b == x.b)
            assert not (a == x.b and b == x.a)

        name = f"C_{a.name}_{b.name}"

        self._connections.append(
            connection_builder(
                f"C_{a.name}_{b.name}", a, b, jnp.array(area), jnp.array(force)
            )
        )

    def external_force(self, a: point_builder, force: Array):
        self._ck_point(a)
        _ = convert_unit(force, self._default_force)
        self._external_forces.append((a, force))


##########################################


class connection(eqx.Module):
    a: str = eqx.field(static=True)
    b: str = eqx.field(static=True)
    area: Array
    # > 0 ==> compression
    force: Array


class graph(eqx.Module):
    points: dict[str, Array]
    connections: dict[str, connection]
    external_forces: dict[str, Array]

    def forces_aggregate(self) -> dict[str, Array]:
        ans = self.external_forces.copy()
        for c in self.connections.values():
            v = self.points[c.b] - self.points[c.a]
            v_len = jnp.linalg.norm(v)
            v_dir = v / v_len

            force_vec_on_a = -c.force * v_dir

            ans[c.a] += force_vec_on_a
            ans[c.b] -= force_vec_on_a
        return ans

    # def distance_based_forces(self):
    #     point_forces: dict[str, Array] = {
    #         x: jnp.zeros_like(y) for x, y in self.points.items()
    #     }

    #     def addforce(p: str, vec: Array):
    #         point_forces[p] += vec

    #     for c in self.connections.values():
    #         # print(c.a.coord)
    #         # print(c.b)
    #         # for x in c.b.coords:
    #         #     print(type(x))
    #         #     print(x)
    #         # print()
    #         # print()
    #         # print()

    #         v = self.points[c.b] - self.points[c.a]
    #         v_len = jnp.linalg.norm(v)
    #         v_dir = v / v_len

    #         # prop_delta = (v_len - get_primal(v_len)) / v_len
    #         prop_delta = (v_len - lax.stop_gradient(v_len)) / v_len

    #         # prop_delta > 0 ==> tension
    #         new_force_val = -prop_delta * c.area * ctx.modulus_of_elasticity
    #         # new_force_val = (
    #         #     c.force - prop_delta * c.area * self._ctx.modulus_of_elasticity
    #         # )

    #         force_vec = new_force_val * v_dir
    #         check_unit(force_vec, ctx.structural_pound)

    #         addforce(c.a, -force_vec)
    #         addforce(c.b, force_vec)

    #     return point_forces
