import warp as wp


def init_liquid(
    particles: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    init_nx: int,
    init_ny: int,
    init_nz: int,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    n_particles: int,
    dx: float,
    kernel_scale: int,
) -> wp.array(dtype=wp.vec3):
    """
    return the boundary_particles
    """
    assert particles.shape == (n_particles,)
    assert velocities.shape == (n_particles,)

    @wp.kernel
    def _init_fluid_particles(
        particles: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        ny: int,
        nz: int,
        dx: float,
    ):
        i, j, k = wp.tid()
        p_idx = i * ny * nz + j * nz + k  # particle index

        p = dx * wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)) + wp.vec3(
            5.0, 5.0, 50.0
        )
        particles[p_idx] = p

        velocities[p_idx] = wp.vec3(0.0, 0.0, 0.0)

    @wp.kernel
    def _init_boundary_particles(
        boundary_particles: wp.array(dtype=wp.vec3),
        nx: int,
        ny: int,
        nz: int,
        single_xy_middle_layer: int,
        dx: float,
        kernel_scale: int,
    ):
        i, j, k = wp.tid()
        if (
            i >= kernel_scale
            and i < nx + kernel_scale
            and j >= kernel_scale
            and j < ny + kernel_scale
            and k >= kernel_scale
            and k < nz + kernel_scale
        ):
            return

        p = (
            wp.vec3(
                wp.float32(i - kernel_scale),
                wp.float32(j - kernel_scale),
                wp.float32(k - kernel_scale),
            )
            * dx
        )
        index = 0
        if k < kernel_scale:
            index = i * (ny + 2 * kernel_scale) * kernel_scale + j * kernel_scale + k
        elif k >= kernel_scale and k < nz + kernel_scale:
            base = kernel_scale * (ny + 2 * kernel_scale) * (nx + 2 * kernel_scale)
            if j < kernel_scale:
                index = (
                    base
                    + (k - kernel_scale) * single_xy_middle_layer
                    + j * (nx + 2 * kernel_scale)
                    + i
                )
            elif j >= kernel_scale and j < ny + kernel_scale:
                base1 = (nx + 2 * kernel_scale) * kernel_scale
                if i < kernel_scale:
                    index = (
                        base
                        + (k - kernel_scale) * single_xy_middle_layer
                        + base1
                        + (j - kernel_scale) * 2 * kernel_scale
                        + i
                    )
                else:
                    index = (
                        base
                        + (k - kernel_scale) * single_xy_middle_layer
                        + base1
                        + (j - kernel_scale) * 2 * kernel_scale
                        + i
                        - nx
                    )
            else:
                base1 = (nx + 2 * kernel_scale) * kernel_scale + 2 * kernel_scale * ny
                index = (
                    base
                    + (k - kernel_scale) * single_xy_middle_layer
                    + base1
                    + ((j - kernel_scale - ny) * (nx + 2 * kernel_scale) + i)
                )
        else:
            base = (
                kernel_scale * (ny + 2 * kernel_scale) * (nx + 2 * kernel_scale)
                + single_xy_middle_layer * nz
            )
            index = (
                base
                + i * (ny + 2 * kernel_scale) * kernel_scale
                + j * kernel_scale
                + (k - kernel_scale - nz)
            )

        boundary_particles[index] = p

    single_layer = (grid_nx + 2 * kernel_scale) * (
        grid_ny + 2 * kernel_scale
    ) - grid_nx * grid_ny
    nb_particles = (
        2 * kernel_scale * ((grid_nx + 2 * kernel_scale) * (grid_ny + 2 * kernel_scale))
        + single_layer * grid_nz
    )
    assert (
        nb_particles
        == (
            (grid_nx + 2 * kernel_scale)
            * (grid_ny + 2 * kernel_scale)
            * (grid_nz + 2 * kernel_scale)
        )
        - grid_nx * grid_ny * grid_nz
    )
    new_boundary_particles = wp.zeros(
        shape=nb_particles,
        dtype=wp.vec3,
    )

    wp.launch(
        _init_fluid_particles,
        dim=(init_nx, init_ny, init_nz),
        inputs=[
            particles,
            velocities,
            init_ny,
            init_nz,
            dx,
        ],
    )

    wp.launch(
        _init_boundary_particles,
        dim=(
            grid_nx + 2 * kernel_scale,
            grid_ny + 2 * kernel_scale,
            grid_nz + 2 * kernel_scale,
        ),
        inputs=[
            new_boundary_particles,
            grid_nx,
            grid_ny,
            grid_nz,
            single_layer,
            dx,
            kernel_scale,
        ],
    )

    return new_boundary_particles, nb_particles
