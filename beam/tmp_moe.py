from lib.utils import allow_autoreload
from pintax import areg, convert_unit, unitify


@allow_autoreload
@unitify
def main():
    len_tot = 21 * areg.inch
    fail_w = 175.0 * areg.force_gram

    b = 1 / 4 * areg.inch
    h = 1 / 8 * areg.inch

    S = b * h**2 / 6

    stress = 6000 * areg.psi

    fail_mom = len_tot / 2 * fail_w / 2 * 2
    max_mom = stress * S

    # max_mom = moe * stress * S
    # moe = fail_mom / stress / S

    print(fail_mom, convert_unit(max_mom, fail_mom))

    # s = b /
