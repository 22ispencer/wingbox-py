import math
from numba import guvectorize, vectorize, float64
import numpy as np


@guvectorize("(n)->(n)", nopython=True)
def section_properties(stringer_locs, res):
    str_count = stringer_locs.shape[0]
    props = np.zeros((str_count + 1, 6))
    props[0, 0] = 0.50122000000  # Area
    props[0, 1] = 1.75840199900  # y
    props[0, 2] = 0.75575764850  # z
    props[0, 3] = 0.11119666770  # Iyy
    props[0, 4] = 1.24456607500  # Izz
    props[0, 5] = 0.08223699025  # Iyz

    # Stringer properties
    props[1:, 0] = 0.05625
    props[1:, 1:2] = 0.0000203451  # Iyy & Izz are identical Iyz = 0

    for i in range(stringer_locs.shape[0]):
        if stringer_locs[i] < 4:
            props[1 + i, 1] = (
                15 * math.cos(0.066865794705 * stringer_locs[i] + 1.43706473738) + 2
            )
            props[1 + i, 1] = (
                15 * math.sin(0.066865794705 * stringer_locs[i] + 1.43706473738)
                - 13.6160687473
            )
        else:
            props[1 + i, 1] = 8 - stringer_locs[i]
            props[1 + i, 2] = props[1 + i, 1] / 8

    y_bar = np.sum(props[:, 0] * props[:, 1]) / np.sum(props[:, 0])
    z_bar = np.sum(props[:, 0] * props[:, 2]) / np.sum(props[:, 0])
    # Iyy
    res[0] = np.sum(props[:, 3]) + np.sum(props[:, 0] * (props[:, 2] - z_bar))
    # Izz
    res[1] = np.sum(props[:, 4]) + np.sum(props[:, 0] * (props[:, 1] - y_bar))
    # Iyz
    res[2] = np.sum(props[:, 5]) + np.sum(
        props[:, 0] * (props[:, 1] - y_bar) * (props[:, 2] - z_bar)
    )
    res[3] = y_bar


@vectorize([float64(float64, float64, float64, float64, float64)], nopython=True)
def deflection_y(mod_elastic, i_yy, i_zz, i_yz, load):
    return (i_yz / (mod_elastic * (i_yy * i_zz - i_yz**2))) * (load - 5) * -32180.83333


@vectorize([float64(float64, float64, float64, float64, float64)], nopython=True)
def deflection_z(mod_elastic, i_yy, i_zz, i_yz, load):
    return (i_zz / (mod_elastic * (i_yy * i_zz - i_yz**2))) * (load - 5) * -32180.83333


@vectorize(nopython=True)
def torsion(x, load, y_bar):
    return 0.00018401233 * x * shear_flow(load, y_bar)


@vectorize([float64(float64, float64)], nopython=True)
def shear_flow(load, y_bar):
    return (load * (y_bar - 1) + 5 * y_bar - 10) / 4.136


@vectorize([float64(float64, float64, float64)], nopython=True)
def stress_shear(load, y_bar, thickness):
    return (load * (y_bar - 1) + 5 * y_bar - 10) / (4.136 * thickness)


@vectorize(
    [float64(float64, float64, float64, float64, float64, float64, float64)],
    nopython=True,
)
def stress_normal(
    x,
    y,
    z,
    load,
    i_yy,
    i_zz,
    i_yz,
):
    return -y * ((load - 5) * (x - 45.75) * i_yy / (i_yy * i_zz - i_yz**2)) + z * (
        (load - 5) * (x - 45.75) * i_zz / (i_yy * i_zz - i_yz**2)
    )


@vectorize(
    [
        float64(
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
    nopython=True,
)
def failed(x, y, z, load, i_yy, i_zz, i_yz, y_bar, thickness):
    shear = stress_shear(load, y_bar, thickness)
    normal = stress_normal(x, y, z, load, i_yy, i_zz, i_yz)

    f_1 = -0.038
    f_11 = 0.0096
    f_66 = 0.44

    if thickness < 1 / 8:
        f_1 = -0.054
        f_11 = 0.0072

    if thickness < 1 / 16:
        f_1 = -0.048
        f_11 = 0.0078
        f_66 = 0.31

    return f_1 * normal + f_11 * normal**2 + f_66 * shear**2 >= 1


@guvectorize([float64[:], float64], "(n)->()")
def count_stacked(places, out):
    _, counts = np.unique(places, return_counts=True)
    out = np.sum(counts) - places.shape[0]


@guvectorize([float64[:], float64], "(n)->()")
def count_adjacent(places, out):
    unique = np.unique(places)
    dup = np.roll(unique, 1)
    out = np.count_nonzero(unique - dup <= 1 / 8)


@vectorize
def final_score(n_stringer, n_ribs, n_stacked, n_adj, load, def_q, def_max, twist):
    weight = 0.4369 + n_ribs * 0.00213 + n_stringer * 0.0146
    design = 100 * (
        0.6 * 8 / (n_stringer + 1)
        + 0.3 * 15 / (n_ribs + 1)
        - (n_stacked / 8 + n_adj / 8)
    )
    perf = (
        0.5 * load / weight
        + 0.5 / def_q
        + 0.05 * (load / def_max + load / twist)
        - 10 * weight
    )
    return design + perf
