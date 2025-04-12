import warp as wp
from util.warp_util import to2d, to3d


def init_leapfrog(
    particles: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec2),
    boundary_particles: wp.array(dtype=wp.vec3),
    init_nx: int,
    init_ny: int,
    dx: float,
    kernel_scale: int,
):
    assert particles.shape == (init_nx * init_ny,)
    assert velocities.shape == (init_nx * init_ny,)
    assert boundary_particles.shape == (
        (init_nx + 2 * kernel_scale) * (init_ny + 2 * kernel_scale) - init_nx * init_ny,
    )

    c1 = wp.constant(wp.vec2(0.25, 0.62))
    c2 = wp.constant(wp.vec2(0.25, 0.38))
    c3 = wp.constant(wp.vec2(0.25, 0.74))
    c4 = wp.constant(wp.vec2(0.25, 0.26))
    w1 = wp.constant(-0.5)
    w2 = wp.constant(0.5)
    w3 = wp.constant(-0.5)
    w4 = wp.constant(0.5)

    @wp.func
    def angular_vel_func(r: float, rad: float = 0.02, strength: float = -0.01) -> float:
        r = r + 1e-6
        linear_vel = strength * 1.0 / r * (1.0 - wp.exp(-(r**2.0) / (rad**2.0)))
        return 1.0 / r * linear_vel

    @wp.kernel
    def _init_leapfrog_particles(
        particles: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec2),
        ny: int,
        dx: float,
    ):
        i, j = wp.tid()
        p_idx = i * ny + j  # particle index

        p = wp.vec2(wp.float32(i), wp.float32(j)) * dx
        particles[p_idx] = to3d(p)

        velocities[p_idx] = wp.vec2()

        diff = p - c1
        r = wp.length(diff)
        velocities[p_idx] += (
            angular_vel_func(r, 0.02, -0.01) * w1 * wp.vec2(-diff.y, diff.x)
        )

        diff = p - c2
        r = wp.length(diff)
        velocities[p_idx] += (
            angular_vel_func(r, 0.02, -0.01) * w2 * wp.vec2(-diff.y, diff.x)
        )

        diff = p - c3
        r = wp.length(diff)
        velocities[p_idx] += (
            angular_vel_func(r, 0.02, -0.01) * w3 * wp.vec2(-diff.y, diff.x)
        )

        diff = p - c4
        r = wp.length(diff)
        velocities[p_idx] += (
            angular_vel_func(r, 0.02, -0.01) * w4 * wp.vec2(-diff.y, diff.x)
        )

    @wp.kernel
    def _init_leapfrog_boundary_particles(
        boundary_particles: wp.array(dtype=wp.vec3),
        nx: int,
        ny: int,
        dx: float,
        kernel_scale: int,
    ):
        i, j = wp.tid()
        if (
            i >= kernel_scale
            and i < nx + kernel_scale
            and j >= kernel_scale
            and j < ny + kernel_scale
        ):
            return

        p = (
            wp.vec2(
                wp.float32(i - kernel_scale),
                wp.float32(j - kernel_scale),
            )
            * dx
        )
        index = 0
        if i < kernel_scale:
            index = i * (ny + 2 * kernel_scale) + j
        elif i >= kernel_scale and i < nx + kernel_scale:
            base = kernel_scale * (ny + 2 * kernel_scale)
            if j < kernel_scale:
                index = base + (i - kernel_scale) * 2 * kernel_scale + j
            else:
                index = base + (i - kernel_scale) * 2 * kernel_scale + j - ny
        else:
            base = (kernel_scale + nx) * (ny + 2 * kernel_scale) - nx * ny
            index = base + (i - kernel_scale - nx) * (ny + 2 * kernel_scale) + j

        boundary_particles[index] = to3d(p)

    wp.launch(
        _init_leapfrog_particles,
        dim=(init_nx, init_ny),
        inputs=[
            particles,
            velocities,
            init_ny,
            dx,
        ],
    )

    wp.launch(
        _init_leapfrog_boundary_particles,
        dim=(init_nx + 2 * kernel_scale, init_ny + 2 * kernel_scale),
        inputs=[
            boundary_particles,
            init_nx,
            init_ny,
            dx,
            kernel_scale,
        ],
    )
