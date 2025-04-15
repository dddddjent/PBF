import warp as wp
from typing import Any, Tuple

wp.set_module_options({"max_unroll": 2})


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
def dN_2(x: float) -> float:
    result = 0.0
    abs_x = wp.abs(x)
    if abs_x < 0.5:
        result = -2.0 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2.0 * abs_x - 3.0)
    if x < 0.0:  # if x < 0 then abs_x is -x
        result *= -1.0
    return result


@wp.func
def sample(
    qf: wp.array3d(dtype=wp.float32), u: int, v: int, w: int, dim: wp.vec3i
) -> float:
    i = wp.max(0, wp.min(u, dim.x - 1))
    j = wp.max(0, wp.min(v, dim.y - 1))
    k = wp.max(0, wp.min(w, dim.z - 1))
    return qf[i, j, k]


# BL_x, BL_y, BL_z are the origin of the grid
@wp.func
def interp_2(
    grid: wp.array3d(dtype=Any),
    p: wp.vec3,
    dim: wp.vec3i,
    BL_x: float = 0.0,
    BL_y: float = 0.0,
    BL_z: float = 0.0,
    dx: float = 1.0,
) -> float:
    u = p.x / dx - BL_x
    v = p.y / dx - BL_y
    w = p.z / dx - BL_z
    s = wp.max(1.0, wp.min(u, wp.float32(dim.x) - 2.0 - 1e-7))
    t = wp.max(1.0, wp.min(v, wp.float32(dim.y) - 2.0 - 1e-7))
    l = wp.max(1.0, wp.min(w, wp.float32(dim.z) - 2.0 - 1e-7))

    # floor
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(l))

    interped = wp.float32(0.0)

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


# BL_x, BL_y, BL_z are the origin of the grid
@wp.func
def interp_2_grad(
    grid: wp.array3d(dtype=Any),
    p: wp.vec3,
    dim: wp.vec3i,
    BL_x: float = 0.0,
    BL_y: float = 0.0,
    BL_z: float = 0.0,
    dx: float = 1.0,
) -> Tuple[float, wp.vec3]:
    u = p.x / dx - BL_x
    v = p.y / dx - BL_y
    w = p.z / dx - BL_z
    s = wp.max(1.0, wp.min(u, wp.float32(dim.x) - 2.0 - 1e-7))
    t = wp.max(1.0, wp.min(v, wp.float32(dim.y) - 2.0 - 1e-7))
    l = wp.max(1.0, wp.min(w, wp.float32(dim.z) - 2.0 - 1e-7))

    # floor
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(l))

    partial_x = wp.float32(0.0)
    partial_y = wp.float32(0.0)
    partial_z = wp.float32(0.0)
    interped = wp.float32(0.0)

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - wp.float32(iu + i)  # x_p - x_i
                y_p_y_i = t - wp.float32(iv + j)
                z_p_z_i = l - wp.float32(iw + k)
                value = sample(grid, iu + i, iv + j, iw + k, dim)
                partial_x += (
                    1.0 / dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i))
                )
                partial_y += (
                    1.0 / dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i))
                )
                partial_z += (
                    1.0 / dx * (value * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i))
                )
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)

    return interped, wp.vec3(partial_x, partial_y, partial_z)


@wp.kernel
def enforce_boundary_SDF(
    particles: wp.array(dtype=wp.vec3),
    boundary: wp.vec3,
    SDF: wp.array3d(dtype=wp.float32),
):
    i = wp.tid()
    p = particles[i]
    p.x = wp.max(0.0, wp.min(boundary.x, p.x))
    p.y = wp.max(0.0, wp.min(boundary.y, p.y))
    p.z = wp.max(0.0, wp.min(boundary.z, p.z))

    dim = wp.vec3i(SDF.shape[0], SDF.shape[1], SDF.shape[2])
    phi, grad = interp_2_grad(SDF, p, dim)
    if phi < 0.0 - 1e-6:
        p = p - (1.0 + 1e-3) * wp.normalize(grad) * phi

    particles[i] = p
