import itertools
import numpy as np

import solve

MIN_STRINGER_COUNT = 5
MAX_STRINGER_COUNT = 10
STRINGER_PLACEMENT_STEP = 1 / 8


def cross_sections(min_stringer_count: int, max_stringer_count: int):
    positions = np.arange(0, 8, STRINGER_PLACEMENT_STEP)

    for stringer_num in range(min_stringer_count, max_stringer_count + 1):
        yield itertools.combinations(positions, r=stringer_num)


if __name__ == "__main__":
    num = 5
    for x_sections in cross_sections(num, num):
        cs_array = np.array(list(x_sections))
        props = np.zeros((cs_array.shape[0], cs_array.shape[1]))
        solve.section_properties(cs_array, props)

        step = 50

        loads = np.full(props.shape[0], 100.0)

        failed = []

        for i in range(6):
            failed = solve.failed(
                0,
                0,
                0,
                loads,
                props[:, 0],
                props[:, 1],
                props[:, 2],
                props[:, 3],
                1 / 8,
            )

            loads = np.where(failed, loads - step, loads + step)
            step /= 2

        loads = np.where(failed, loads - step, loads)
        print(loads)

        adj = np.zeros(cs_array.shape[0])
        solve.count_adjacent(cs_array, adj)

        stacked = np.zeros(cs_array.shape[0])
        solve.count_stacked(cs_array, stacked)

        n_ribs = 15
        n_stringers = cs_array.shape[1]

        def_q = solve.deflection_z(awq)
