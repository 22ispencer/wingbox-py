from numba import float64, vectorize


# Internal Loads
@vectorize(float64(float64, float64))
def moment_y(P, x):
    return (P - 5) * (x - 45.75)
