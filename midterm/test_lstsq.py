import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.random import PRNGKey

from lib.lstsq import lstsq_safe

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

np.set_printoptions(precision=10, suppress=True)


def main():
    key = PRNGKey(0)
    k1, k2, k3, k4 = random.split(key, num=4)

    s = 0.000000001

    a = random.uniform(k1, (100, 100))
    b = random.uniform(k2, (100,))

    ans1 = lstsq_safe(a, b)
    ans2, _, _, _ = jnp.linalg.lstsq(a, b)
    return ans1, ans1 - ans2

    ad = random.uniform(k3, (100, 100))
    # ad = jnp.zeros_like(a)
    # bd = random.uniform(k4, (100,))
    bd = jnp.zeros_like(b)

    primals_out, tangents_out = jax.jvp(lstsq_safe, (a, b), (ad, bd))

    ans = (lstsq_safe(a + s * ad, b + s * bd) - primals_out) / s

    return ans, tangents_out - ans
