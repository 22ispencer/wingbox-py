import pytest
import wingbox_py.geometry as geometry
import numpy as np


def test_rect_pmoa():
    rect = geometry.Rect(0, 0, 6, 2)

    assert rect.smoa() == 4
    assert rect.smoa(np.deg2rad(90)) == 36
    assert pytest.approx(rect.smoa(np.deg2rad(24)), 0.00001) == 9.29391
    assert pytest.approx(rect.smoa(np.deg2rad(72), 35), 0.1) == 14732.9
