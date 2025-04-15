import warp as wp
from typing import Any


# quadratic
@wp.func
def N_2(x: float) -> float:
    result = 0.0
    abs_x = wp.abs(x)
    if abs_x < 0.5:
        result = 3.0 / 4.0 - abs_x**2.0
    elif abs_x < 1.5:
        result = 0.5 * (3.0 / 2.0 - abs_x) ** 2.0
    return result


@wp.func
def sample(
    qf: wp.array3d(dtype=wp.float32), u: int, v: int, w: int, dim: wp.vec3i
) -> float:
    # i = wp.max(0, wp.min(int(u), dim.x - 1))
    # j = wp.max(0, wp.min(int(v), dim.y - 1))
    # k = wp.max(0, wp.min(int(w), dim.z - 1))
    # return qf[i, j, k]
    if u < 0 or v < 0 or w < 0:
        return 1.0
    if u > dim.x - 1 or v > dim.y - 1 or w > dim.z - 1:
        return 1.0
    return qf[int(u), int(v), int(w)]


# assume the grid's spacing is 1.0
# BL_x, BL_y, BL_z are the origin of the grid
@wp.func
def interp_2(
    grid: wp.array3d(dtype=Any),
    p: wp.vec3,
    dim: wp.vec3i,
    BL_x: float = 0.0,
    BL_y: float = 0.0,
    BL_z: float = 0.0,
) -> float:
    u = p.x - BL_x
    v = p.y - BL_y
    w = p.z - BL_z
    s = wp.max(1.0, wp.min(u, wp.float32(dim.x) - 2.0 - 1e-7))
    t = wp.max(1.0, wp.min(v, wp.float32(dim.y) - 2.0 - 1e-7))
    l = wp.max(1.0, wp.min(w, wp.float32(dim.z) - 2.0 - 1e-7))

    # floor
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(l))

    interped = 0.0

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - wp.float32(iu + i)  # x_p - x_i
                y_p_y_i = t - wp.float32(iv + j)
                z_p_z_i = l - wp.float32(iw + k)
                value = sample(grid, iu + i, iv + j, iw + k, dim)
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)

    return interped

