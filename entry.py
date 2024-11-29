import itertools
import numpy as np

MIN_STRINGER_COUNT = 5
MAX_STRINGER_COUNT = 8
STRINGER_PLACEMENT_STEP = 1 / 8


def cross_sections(min_stringer_count: int, max_stringer_count: int):
    positions = np.arange(0, 8, STRINGER_PLACEMENT_STEP)

    cross_sections = []
    for stringer_num in range(min_stringer_count, max_stringer_count):
        combinations = np.array(itertools.combinations(positions, r=stringer_num))
        print(combinations.size)
        cross_sections.append(combinations)

    return cross_sections


if __name__ == "__main__":
    cross_sections(MIN_STRINGER_COUNT, MAX_STRINGER_COUNT)
