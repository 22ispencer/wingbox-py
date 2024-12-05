from datetime import datetime
import itertools
import numpy as np

import solve

MIN_STRINGER_COUNT = 5
MAX_STRINGER_COUNT = 5
STRINGER_PLACEMENT_STEP = 1 / 8


def cross_sections(min_stringer_count: int, max_stringer_count: int):
    positions = np.arange(0, 8, STRINGER_PLACEMENT_STEP)

    for stringer_num in range(min_stringer_count, max_stringer_count + 1):
        yield itertools.combinations(positions, r=stringer_num)


if __name__ == "__main__":
    points = np.load("points.npy")  # t, y, z, thick, E
    f = open("out.txt", "w+")
    for x_sections in cross_sections(MIN_STRINGER_COUNT, MAX_STRINGER_COUNT):
        cs_array = np.array(list(x_sections))
        props = np.zeros((cs_array.shape[0], cs_array.shape[1]))
        solve.section_properties(cs_array, props)

        step = 50

        loads = np.full(props.shape[0], 100.0)

        failed = []

        for i in range(6):
            failed = np.full(loads.shape, False)
            for point in points:
                did_it_fail = solve.failed(
                    0,
                    point[1] - props[:, 3],
                    point[2] - props[:, 4],
                    point[4],
                    loads,
                    props[:, 0],
                    props[:, 1],
                    props[:, 2],
                    props[:, 3],
                    point[3],
                )
                failed = failed | did_it_fail

            loads = np.where(failed, loads - step, loads + step)
            step /= 2

        loads = np.where(failed, loads - step, loads)

        adj = np.zeros(cs_array.shape[0])
        solve.count_adjacent(cs_array, adj)

        stacked = np.zeros(cs_array.shape[0])

        n_ribs = 15
        n_stringers = cs_array.shape[1]
        stacked = 0
        adjacent = 0
        def_q = solve.deflection_y(props[:, 0], props[:, 1], props[:, 2], 0)
        def_max = solve.deflection_y(props[:, 0], props[:, 1], props[:, 2], loads)
        twist = solve.twist(loads, props[:, 3])

        scores = solve.final_score(
            n_stringers, n_ribs, stacked, adjacent, loads, def_q, def_max, twist
        )
        max_ind = np.argmax(scores)
        report = (
            f"{n_stringers} Stringers\n"
            "-----------------------\n"
            f"Score: {scores[max_ind]}\n"
            f"Shear deflection (twist): {twist[max_ind]}\n"
            f"Q deflection: {def_q[max_ind]}\n"
            f"Max deflection: {def_max[max_ind]}\n"
            f"Max Load: {loads[max_ind]}\n"
            f"Cross-section components: {cs_array[max_ind]}\n"
            f"Section properties (Iyy, Izz, Iyz, y_bar, z_bar):\n"
            f"{props[max_ind,:5]}\n"
            f"{datetime.now()}"
            "\n"
        )
        print(report)
        f.write(report)
    f.close()
