import warp as wp


def init_leapfrog(
    particles: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    boundary_particles: wp.array(dtype=wp.vec2),
    init_nx: int,
    init_ny: int,
    dx: float,
    kernel_scale: int,
    kernel_radius: float,
):
    assert particles.shape == (init_nx * init_ny,)
    assert velocities.shape == (init_nx * init_ny,)
    assert boundary_particles.shape == (
        (init_nx + 2 * kernel_scale) * (init_ny + 2 * kernel_scale) - init_nx * init_ny,
    )

    @wp.kernel
    def _init_leapfrog(
        particles: wp.array(dtype=wp.vec2),
        velocities: wp.array(dtype=wp.vec2),
        nx: int,
        ny: int,
        dx: float,
        kernel_scale: int,
        kernel_radius: float,
    ):
        i, j = wp.tid()
        x = wp.float32(i) * dx
        y = wp.float32(j) * dx
