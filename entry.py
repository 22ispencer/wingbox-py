import itertools
import numpy as np

MIN_STRINGER_COUNT = 5
MAX_STRINGER_COUNT = 10
STRINGER_PLACEMENT_STEP = 1 / 8


def cross_sections(min_stringer_count: int, max_stringer_count: int):
    positions = np.arange(0, 8, STRINGER_PLACEMENT_STEP)

    for stringer_num in range(min_stringer_count, max_stringer_count + 1):
        for c in itertools.combinations(positions, r=stringer_num):
            yield c


if __name__ == "__main__":
    count = 0
    for batch in itertools.batched(
        cross_sections(MIN_STRINGER_COUNT, MAX_STRINGER_COUNT), n=1024
    ):
        count += len(batch)
    print(count)
