from datetime import datetime
import json
import itertools
import numpy as np
import os
import solve

MIN_STRINGER_COUNT = int(os.environ.get("MIN_STRINGER_COUNT", 5))
MAX_STRINGER_COUNT = int(os.environ.get("MAX_STRINGER_COUNT", 5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 10_000_000))
STRINGER_PLACEMENT_STEP = 1 / 8
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "out.json")


def cross_sections(min_stringer_count: int, max_stringer_count: int):
    positions = np.arange(0, 8, STRINGER_PLACEMENT_STEP)

    batches = []
    for stringer_num in range(min_stringer_count, max_stringer_count + 1):
        batches.append(
            itertools.batched(
                itertools.combinations(positions, r=stringer_num), n=BATCH_SIZE
            )
        )
    return itertools.chain(*batches)


if __name__ == "__main__":
    if os.path.isfile(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a") as f:
        f.write("[\n")
    points = np.load("points.npy")  # t, o, z, thick, E
    for x_sections in cross_sections(MIN_STRINGER_COUNT, MAX_STRINGER_COUNT):
        cs_array = np.array(list(x_sections))
        print(cs_array.shape[0])
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

        valid_scores = scores[loads >= 15.0]
        if valid_scores.shape[0] < 1:
            continue
        vmax = np.max(valid_scores)
        max_ind = np.where(scores == vmax)[0][0]
        print(max_ind)
        report_data = {
            "score": float(scores[max_ind]),
            "twist": float(twist[max_ind]),
            "q_def": float(def_q[max_ind]),
            "max_def": float(def_max[max_ind]),
            "load": float(loads[max_ind]),
            "cross_section_locs": cs_array[max_ind].tolist(),
            "section_props": props[max_ind, :5].tolist(),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
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
            "\n\n"
        )
        print(report)
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(report_data) + ",\n")
    with open(OUTPUT_FILE, "a") as f:
        f.write("]\n")
