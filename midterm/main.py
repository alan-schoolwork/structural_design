import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array, lax
from jax import tree_util as jtu
from jax.experimental import sparse
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pintax import areg, convert_unit, dimensionless, magnitude, quantity, unitify

from lib.batched import (
    batched,
    batched_vmap,
    batched_zip,
)
from lib.checkify import checkify_simple
from lib.graph import graph_t, point
from lib.jax_utils import debug_print
from lib.lstsq import flstsq, flstsq_r
from lib.utils import (
    cast,
    cast_unchecked,
    fval,
    jit,
    tree_at_2_,
    value_and_grad_aux_,
)
from midterm.build_graph import build_graph

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

np.set_printoptions(precision=3, suppress=True)


@jit
def solve_forces(g: graph_t):
    connnection_forces = g._connections.map(lambda _: jnp.array(0.0) * areg.pound)

    def get_eqs(connnection_forces=connnection_forces):
        aggr = g.sum_annotations(
            g._points.map(lambda _: jnp.zeros((3,))),
            g.forces_aggregate(connnection_forces, density=1.0 * areg.pound / areg.m),
        )
        aggr_filtered = batched_vmap(
            lambda p, f: lax.select(
                p.accepts_force, on_true=jnp.zeros_like(f), on_false=f
            ),
            g._points,
            aggr,
        )
        return aggr_filtered

    return flstsq(get_eqs, connnection_forces)


state_t = tuple[
    tuple[Array, ...], optax.OptState, flstsq_r[batched[Array], batched[Array]]
]


# @checkify_simple
@unitify
def solve_forces_final():
    g = build_graph()
    print("done building graph")

    return g, solve_forces(g)

    def get_optim_buffers(g: graph_t) -> tuple[Array, ...]:
        return (g._points.unflatten().coords,)

    init_bufs = get_optim_buffers(g)

    def compute_loss(buffers: tuple[Array, ...]):
        cur_g = tree_at_2_(get_optim_buffers, g, buffers)
        cur_g = cur_g.maybe_pin_points()
        ans = solve_forces(cur_g)

        softplus = lambda x: jax.nn.softplus(quantity(x).m) * quantity(x).u
        fs_abs = ans.x.map(lambda f: softplus(f) + softplus(-f))

        loss = jnp.sum(fs_abs.unflatten())
        loss = loss * areg.meter / areg.pound
        return loss, ans

    # (loss, ans), grads = jit(lambda: value_and_grad_aux_(compute_loss)(init_bufs))()
    # print("info:", loss, jnp.max(grads[0]), jnp.min(grads[0]))
    # return g, ans

    optimizer = optax.adam(learning_rate=10 ** (-4))
    init_opt_state = optimizer.init(init_bufs)

    def optim_loop(state: state_t, _) -> tuple[state_t, None]:
        debug_print("optim_loop: starting")
        buffers, opt_state, _ = state
        (loss, ans), grads = value_and_grad_aux_(compute_loss)(buffers)
        debug_print("info:", loss, jnp.max(grads[0]), jnp.min(grads[0]))
        updates, opt_state = optimizer.update(
            tuple(convert_unit(x, dimensionless) for x in grads), opt_state
        )
        buffers = optax.apply_updates(
            tuple(magnitude(x, areg.m) for x in buffers), updates
        )
        buffers = tuple(x * areg.m for x in cast_unchecked()(buffers))
        return (buffers, opt_state, ans), None

    @jit
    def get_ans_as_zero():
        _, ans = compute_loss(init_bufs)
        return jtu.tree_map(lambda x: jnp.zeros_like(x), ans)

    (buffers, opt_state, ans), _ = jit(
        lambda: lax.scan(
            optim_loop,
            init=(init_bufs, init_opt_state, get_ans_as_zero()),
            length=10,
        )
    )()
    return tree_at_2_(get_optim_buffers, g, buffers), ans


def do_plot(res_):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    fig = plt.figure(figsize=(18, 8))
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.suptitle("funicular shape", fontsize=32)

    ax = fig.add_subplot(131, projection="3d", computed_zorder=False)
    assert isinstance(ax, Axes3D)
    ax.set_xlim(-20, 10)
    ax.set_ylim(-10, 20)
    ax.set_zlim(-10, 20)
    do_plot_one(ax, res_)

    ax = fig.add_subplot(132, projection="3d", computed_zorder=False)
    assert isinstance(ax, Axes3D)
    ax.set_xlim(-5, 5)
    ax.set_ylim(5, 15)
    ax.set_zlim(0, 10)
    do_plot_one(ax, res_)

    ax = fig.add_subplot(133, projection="3d", computed_zorder=False)
    assert isinstance(ax, Axes3D)
    ax.set_xlim(8, 23)
    ax.set_ylim(5, 20)
    ax.set_zlim(-17, -2)
    do_plot_one(ax, res_)
    # plt.show()

    plt.tight_layout()
    plt.savefig("funicular1.png", dpi=300)
    plt.close()


def do_plot_one(ax, res_):
    return unitify(lambda res: do_plot_one_(ax, res))(res_)


def do_plot_one_(ax, res_):
    ax.set_axis_off()
    res = cast_unchecked.from_fn(solve_forces_final)(res_)
    # assert False
    g, ans = res
    forces = ans.x
    forces_errors = ans.errors

    # lim = 30
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)

    f_max = jnp.max(jnp.abs(forces.unflatten()))
    print("f_max", f_max)
    f_max = 500 * areg.pound

    def _color(x: fval, end: Array):
        x = lax.min(x / f_max * 10, 1.0)
        return x * end + (1 - x) * jnp.array([1.0, 1.0, 1.0]) * 0.5

    colors = forces.map(
        lambda x: lax.select(
            x > 0,
            on_true=_color(x, jnp.array([1.0, 0.0, 0.0])),
            on_false=_color(-x, jnp.array([0.0, 0.0, 1.0])),
        )
    )
    linewidths = forces.map(lambda x: (jnp.abs(x) / f_max * 10 + 0.2))

    line_collection = Line3DCollection(
        (g.get_lines() / areg.m).tolist(),
        colors=colors.unflatten().tolist(),
        linewidths=linewidths.unflatten().tolist(),
        zorder=1,
    )
    ax.add_collection3d(line_collection)
    # ax.plot(xs, ys, zs)

    # fixed_points, ct = g._points.filter(lambda x: x.fixed)
    plot_errors = batched_zip(g._points, forces_errors)
    plot_errors, ct = plot_errors.filter_arr(
        plot_errors.tuple_map(lambda p, e: jnp.linalg.norm(e) > 0.2 * areg.pound)
    )

    def _plot_errors(x: point, e: Array):
        cd = x.coords / areg.m
        v = e / areg.pound * 2
        return jnp.stack([cd, cd + v]), cd + v, jnp.maximum(jnp.linalg.norm(v), 0.2)

    print("count:", ct)
    ct = int(ct)
    if ct > 0:
        plot_error_lines, plot_error_points, intensity = (
            plot_errors[:ct].tuple_map(_plot_errors).unflatten()
        )
        line_collection = Line3DCollection(
            plot_error_lines.tolist(),
            colors=(0.0, 1.0, 0.0),
            linewidths=(intensity * 3.0).tolist(),
            zorder=5,
        )
        ax.add_collection3d(line_collection)

        ax.scatter(
            plot_error_points[:, 0].tolist(),
            plot_error_points[:, 1].tolist(),
            plot_error_points[:, 2].tolist(),  # type: ignore
            color=(0.0, 0.0, 0.0),
            marker="o",
            s=(intensity * 20.0).tolist(),  # type: ignore
            # s=plot_points_s.tolist(),
            zorder=6,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # plt.show()
    # print(ys)


@unitify
def do_plot_simple():
    g = build_graph()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)

    ax.set_xlim(-20, 10)
    ax.set_ylim(-10, 20)
    ax.set_zlim(-10, 20)

    line_collection = Line3DCollection(
        (g.get_lines() / areg.m).tolist(),
    )
    ax.add_collection3d(line_collection)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
