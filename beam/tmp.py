from lib.utils import allow_autoreload
from pintax import areg, convert_unit, unitify


@allow_autoreload
@unitify
def main():
    len_tot = 21 * areg.inch
    fail_w = 175.0 * areg.g

    b = 1 / 4 * areg.inch
    h = 1 / 8 * areg.inch

    S = b * h**2 / 6

    stress = 1000 * areg.psi

    fail_mom = len_tot * fail_w
    max_mom = stress * S

    # max_mom = moe * stress * S
    # moe = fail_mom / stress / S

    print(fail_mom, max_mom)

    # s = b /
