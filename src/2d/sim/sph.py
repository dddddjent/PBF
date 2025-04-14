import warp as wp
from sim.kernel_function import W, gradW
from util.warp_util import to2d, to3d


def get_initial_density(
    particles: wp.array(dtype=wp.vec3),
    particle_grid: wp.HashGrid,
    boundary_particles: wp.array(dtype=wp.vec3),
    boundary_particle_grid: wp.HashGrid,
    kernel_radius: float,
) -> wp.float32:
    # invoke with 1 thread
    @wp.kernel
    def _get_initial_density(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        kernel_radius: float,
        density: wp.array(dtype=float),
    ):
        p_len = particles.shape[0]
        index = p_len // 2 + 10

        p = particles[index]
        density[0] = wp.float32(0.0)

        query = wp.hash_grid_query(particle_grid, p, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            x_p_neighbor = to2d(p - particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius:
                density[0] += W(x_p_neighbor, kernel_radius) * 1.0

        query = wp.hash_grid_query(boundary_particle_grid, p, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            x_p_neighbor = to2d(p - boundary_particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius:
                density[0] += W(x_p_neighbor, kernel_radius) * 1.0

    density = wp.zeros(1, dtype=float)
    wp.launch(
        _get_initial_density,
        dim=1,
        inputs=[
            particles,
            particle_grid.id,
            boundary_particles,
            boundary_particle_grid.id,
            kernel_radius,
            density,
        ],
    )
    return wp.float32(density.numpy()[0])


@wp.kernel
def forward_euler_advection(
    particles: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec2),
    dt: float,
    particles_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = particles[i]
    v = velocities[i]
    particles_out[i] = to3d(to2d(p) + dt * v)


@wp.kernel
def update_density(
    particles: wp.array(dtype=wp.vec3),
    particle_grid: wp.uint64,
    boundary_particles: wp.array(dtype=wp.vec3),
    boundary_particle_grid: wp.uint64,
    kernel_radius: float,
    densities: wp.array(dtype=float),
):
    i = wp.tid()

    p = particles[i]
    densities[i] = wp.float32(0.0)

    query = wp.hash_grid_query(particle_grid, p, kernel_radius * 2.0)
    query_idx = int(0)
    while wp.hash_grid_query_next(query, query_idx):
        x_p_neighbor = to2d(p - particles[query_idx])
        if wp.length(x_p_neighbor) < kernel_radius * 2.0:
            densities[i] += W(x_p_neighbor, kernel_radius) * 1.0

    query = wp.hash_grid_query(boundary_particle_grid, p, kernel_radius * 2.0)
    query_idx = int(0)
    while wp.hash_grid_query_next(query, query_idx):
        x_p_neighbor = to2d(p - boundary_particles[query_idx])
        if wp.length(x_p_neighbor) < kernel_radius * 2.0:
            densities[i] += W(x_p_neighbor, kernel_radius) * 1.0


@wp.kernel
def apply_acc(
    velocities: wp.array(dtype=wp.vec2),
    acc: wp.array(dtype=wp.vec2),
    dt: float,
):
    """apply gravity here"""
    i = wp.tid()
    velocities[i] += dt * (wp.vec2(0.0, -9.8) + acc[i])


@wp.kernel
def compute_vorticity(
    particles: wp.array(dtype=wp.vec3),
    particle_grid: wp.uint64,
    velocities: wp.array(dtype=wp.vec2),
    kernel_radius: float,
    vorticities: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p = particles[i]

    vorticities[i] = wp.float32(0.0)

    query = wp.hash_grid_query(particle_grid, p, kernel_radius * 2.0)
    query_idx = int(0)
    while wp.hash_grid_query_next(query, query_idx):
        x_p_neighbor = to2d(p - particles[query_idx])
        if wp.length(x_p_neighbor) < kernel_radius * 2.0:
            grad = to3d(gradW(-x_p_neighbor, kernel_radius))
            vij = to3d(velocities[i] - velocities[query_idx])
            vorticities[i] += wp.cross(grad, vij).z


@wp.kernel
def compute_vorticity_confinement_acc(
    particles: wp.array(dtype=wp.vec3),
    particle_grid: wp.uint64,
    vorticities: wp.array(dtype=wp.float32),
    densities: wp.array(dtype=wp.float32),
    kernel_radius: float,
    acc: wp.array(dtype=wp.vec2),
    strength: float = 1.0,
):
    i = wp.tid()
    p = particles[i]

    eta = wp.vec2(0.0, 0.0)

    query = wp.hash_grid_query(particle_grid, p, kernel_radius * 2.0)
    query_idx = int(0)
    while wp.hash_grid_query_next(query, query_idx):
        x_p_neighbor = to2d(p - particles[query_idx])
        if wp.length(x_p_neighbor) < kernel_radius * 2.0:
            eta += (
                densities[i]
                * (
                    vorticities[i] / (densities[i] ** 2.0)
                    + vorticities[query_idx] / (densities[query_idx] ** 2.0)
                )
                * gradW(x_p_neighbor, kernel_radius)
            )
    N = wp.vec2(0.0, 0.0)
    if wp.length(eta) > 0.0:
        N = wp.normalize(eta)

    f = wp.cross(to3d(N), wp.vec3(0.0, 0.0, vorticities[i]))

    acc[i] += strength * to2d(f) / densities[i]
