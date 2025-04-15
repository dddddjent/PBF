import warp as wp
from sim.kernel_function import W, gradW
from util.warp_util import to2d, to3d
from util.io_util import debug_particle_field, debug_particle_vector_field
import sim.sph as sph


class PBFPossionSolver:
    def __init__(
        self,
        particle_grid: wp.HashGrid,
        boundary_particles: wp.array,
        boundary_particle_grid: wp.HashGrid,
        alphas: wp.array,
        n_particles: int,
        kernel_radius: float,
        d0: float,
        error_tolerance: float,
        max_iterations: int,
        boundary_size: tuple = (256.0, 256.0),
        k_corr: float = -0.1,
        q_corr: float = 0.2,
        n_corr: float = 4.0,
    ):
        self.particle_grid = particle_grid
        self.boundary_particles = boundary_particles
        self.boundary_particle_grid = boundary_particle_grid
        self.alphas = alphas
        self.n_particles = n_particles
        self.kernel_radius = kernel_radius
        self.d0 = d0
        self.error_tolerance = error_tolerance
        self.max_iterations = max_iterations
        self.boundary = wp.vec2(boundary_size[0], boundary_size[1])
        self.corr_parameters = wp.vec3(k_corr, q_corr, n_corr)

        self.lambdas = wp.zeros(n_particles, dtype=wp.float32)
        self.particle_buffer = wp.zeros(n_particles, dtype=wp.vec3)
        self.C = wp.zeros(n_particles, dtype=wp.float32)
        self.gradC = wp.zeros(n_particles, dtype=wp.vec2)
        self.errors = wp.zeros(n_particles, dtype=wp.float32)

    @wp.kernel
    def compute_C(
        densities: wp.array(dtype=wp.float32),
        d0: float,
        C: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        C[i] = densities[i] / d0 - 1.0

    @wp.kernel
    def compute_gradC(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        d0: float,
        kernel_radius: float,
        gradC: wp.array(dtype=wp.vec2),
    ):
        i = wp.tid()
        p = particles[i]

        gradC[i] = wp.vec2(0.0, 0.0)

        query = wp.hash_grid_query(particle_grid, p, kernel_radius * 2.0)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue
            x_p_neighbor = to2d(p - particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius * 2.0:
                gradC[i] += gradW(x_p_neighbor, kernel_radius)

        query = wp.hash_grid_query(boundary_particle_grid, p, kernel_radius * 2.0)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            x_p_neighbor = to2d(p - boundary_particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius * 2.0:
                gradC[i] += gradW(x_p_neighbor, kernel_radius)

        gradC[i] /= d0

    @wp.kernel
    def compute_lambdas(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        C: wp.array(dtype=wp.float32),
        gradC: wp.array(dtype=wp.vec2),
        d0: float,
        kernel_radius: float,
        alphas: wp.array(dtype=wp.float32),
        dt: float,
        lambdas: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        p = particles[i]

        temp = wp.dot(gradC[i], gradC[i])

        query = wp.hash_grid_query(particle_grid, p, kernel_radius * 2.0)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue
            x_p_neighbor = to2d(p - particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius * 2.0:
                grad = -gradW(x_p_neighbor, kernel_radius) / d0
                temp += wp.dot(grad, grad)

        query = wp.hash_grid_query(boundary_particle_grid, p, kernel_radius * 2.0)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            x_p_neighbor = to2d(p - boundary_particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius * 2.0:
                grad = -gradW(x_p_neighbor, kernel_radius) / d0
                temp += wp.dot(grad, grad)

        lambdas[i] = lambdas[i] + (-C[i] - alphas[i] / (dt * dt) * lambdas[i]) / (
            alphas[i] / (dt * dt) + temp
        )

    @wp.kernel
    def update_positions(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        lambdas: wp.array(dtype=wp.float32),
        d0: float,
        kernel_radius: float,
        corr_parameters: wp.vec3,
        particles_out: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()

        p = particles[i]

        delta = wp.vec2(0.0, 0.0)

        query = wp.hash_grid_query(particle_grid, p, kernel_radius * 2.0)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue
            x_p_neighbor = to2d(p - particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius * 2.0:
                s_corr = (
                    corr_parameters.x
                    * (
                        W(x_p_neighbor, kernel_radius)
                        / W(corr_parameters.y * kernel_radius, kernel_radius)
                    )
                    ** corr_parameters.z
                )
                delta += (
                    (lambdas[i] + lambdas[query_idx] + s_corr)
                    * gradW(x_p_neighbor, kernel_radius)
                    / d0
                )

        query = wp.hash_grid_query(boundary_particle_grid, p, kernel_radius * 2.0)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            x_p_neighbor = to2d(p - boundary_particles[query_idx])
            if wp.length(x_p_neighbor) < kernel_radius * 2.0:
                s_corr = (
                    corr_parameters.x
                    * (
                        W(x_p_neighbor, kernel_radius)
                        / W(corr_parameters.y * kernel_radius, kernel_radius)
                    )
                    ** corr_parameters.z
                )
                delta += (
                    (lambdas[i] + lambdas[query_idx] + s_corr)
                    * gradW(x_p_neighbor, kernel_radius)
                    / d0
                )

        particles_out[i] = p + to3d(delta)

    @wp.kernel
    def enforce_boundary(particles: wp.array(dtype=wp.vec3), boundary: wp.vec2):
        i = wp.tid()
        p = particles[i]
        p.x = wp.max(0.0, wp.min(boundary.x, p.x))
        p.y = wp.max(0.0, wp.min(boundary.y, p.y))
        particles[i] = p

    @wp.kernel
    def compute_error(
        densities: wp.array(dtype=wp.float32),
        d0: float,
        errors: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        errors[i] = wp.abs(densities[i] - d0) / d0

    def solve(
        self,
        particles: wp.array,
        densities: wp.array,
        dt: float,
        debug_info: str = "",
    ):
        self.lambdas = wp.zeros(self.n_particles, dtype=wp.float32)

        iter = 0
        error = 10000
        while iter < self.max_iterations and error > self.error_tolerance:
            self.particle_grid.build(points=particles, radius=self.kernel_radius * 2.0)

            wp.launch(
                sph.update_density,
                dim=self.n_particles,
                inputs=[
                    particles,
                    self.particle_grid.id,
                    self.boundary_particles,
                    self.boundary_particle_grid.id,
                    self.kernel_radius,
                    densities,
                ],
            )
            wp.launch(
                self.compute_C,
                dim=self.n_particles,
                inputs=[
                    densities,
                    self.d0,
                    self.C,
                ],
            )
            wp.launch(
                self.compute_gradC,
                dim=self.n_particles,
                inputs=[
                    particles,
                    self.particle_grid.id,
                    self.boundary_particles,
                    self.boundary_particle_grid.id,
                    self.d0,
                    self.kernel_radius,
                    self.gradC,
                ],
            )
            wp.launch(
                self.compute_lambdas,
                dim=self.n_particles,
                inputs=[
                    particles,
                    self.particle_grid.id,
                    self.boundary_particles,
                    self.boundary_particle_grid.id,
                    self.C,
                    self.gradC,
                    self.d0,
                    self.kernel_radius,
                    self.alphas,
                    dt,
                    self.lambdas,
                ],
            )
            wp.launch(
                self.update_positions,
                dim=self.n_particles,
                inputs=[
                    particles,
                    self.particle_grid.id,
                    self.boundary_particles,
                    self.boundary_particle_grid.id,
                    self.lambdas,
                    self.d0,
                    self.kernel_radius,
                    self.corr_parameters,
                    self.particle_buffer,
                ],
            )
            wp.launch(
                self.enforce_boundary,
                dim=self.n_particles,
                inputs=[self.particle_buffer, self.boundary],
            )

            wp.copy(particles, self.particle_buffer, count=self.n_particles)

            wp.launch(
                self.compute_error,
                dim=self.n_particles,
                inputs=[
                    densities,
                    self.d0,
                    self.errors,
                ],
            )
            error = wp.utils.array_sum(self.errors) / self.n_particles
            iter += 1
        wp.launch(
            sph.update_density,
            dim=self.n_particles,
            inputs=[
                particles,
                self.particle_grid.id,
                self.boundary_particles,
                self.boundary_particle_grid.id,
                self.kernel_radius,
                densities,
            ],
        )
        print(f"{debug_info}  iter: {iter}  error: {error}")
