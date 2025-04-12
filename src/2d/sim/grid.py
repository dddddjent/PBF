import warp as wp

from typing import Any


@wp.func
def sample_grid(
    field: wp.array2d(dtype=Any), shape: wp.vec2i, u: Any, v: Any
) -> Any:
    i = wp.max(0, wp.min(int(u), shape[0] - 1))
    j = wp.max(0, wp.min(int(v), shape[1] - 1))
    return field[i, j]


@wp.kernel
def curl(
    grid_velocities: wp.array2d(dtype=wp.vec2),
    grid_vorticities: wp.array2d(dtype=float),
    shape: wp.vec2i,
    dx: float,
):
    i, j = wp.tid()
    vl = sample_grid(grid_velocities, shape, i - 1, j)
    vr = sample_grid(grid_velocities, shape, i + 1, j)
    vb = sample_grid(grid_velocities, shape, i, j - 1)
    vt = sample_grid(grid_velocities, shape, i, j + 1)
    grid_vorticities[i, j] = (vr.y - vl.y - vt.x + vb.x) / (2.0 * dx)
