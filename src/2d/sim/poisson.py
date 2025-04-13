import warp as wp
from sim.kernel_function import W, gradW
from util.warp_util import to2d


# Static boundary conditions
class PoissonSolver:
    def __init__(
        self,
        particle_grid: wp.HashGrid,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.HashGrid,
        n_particles: int,
        kernel_radius: float,
        d0: float,  # initial density
        error_tolerance: float = 1e-4,
        max_iterations: int = 100,
    ):
        self.particle_grid = particle_grid
        self.boundary_particles = boundary_particles
        self.boundary_particle_grid = boundary_particle_grid
        self.n_particles = n_particles
        self.kernel_radius = kernel_radius
        self.d0 = d0
        self.error_tolerance = error_tolerance
        self.max_iterations = max_iterations

        self.aii = wp.zeros(n_particles, dtype=wp.float32)  # a_ii in A
        self.dii = wp.zeros(n_particles, dtype=wp.vec2)
        self.sum_dij_pj = wp.zeros(n_particles, dtype=wp.vec2)
        self.sum_dij_pj_boundary = wp.zeros(boundary_particles.shape[0], dtype=wp.vec2)
        self.s = wp.zeros(n_particles, dtype=wp.float32)
        self.w = wp.constant(wp.float32(0.5))
        self.pressure_tmp = wp.zeros(n_particles, dtype=wp.float32)
        self.err = wp.zeros(n_particles, dtype=wp.float32)

    @wp.kernel
    def _compute_source_term(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        velocities: wp.array(dtype=wp.vec2),
        densities: wp.array(dtype=wp.float32),
        kernel_radius: float,
        d0: float,
        dt: float,
        ignore_density_diff: float,
        s: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        pos = particles[i]

        delta_density = float(0.0)
        query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue

            x_i_neighbor = to2d(pos - particles[query_idx])
            if wp.length(x_i_neighbor) < kernel_radius:
                delta_density += 1.0 * wp.dot(
                    (velocities[i] - velocities[query_idx]),
                    gradW(x_i_neighbor, kernel_radius),
                )

        # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
        # query_idx = int(0)
        # while wp.hash_grid_query_next(query, query_idx):
        #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
        #     if wp.length(x_i_neighbor) < kernel_radius:
        #         delta_density += 1.0 * wp.dot(
        #             (velocities[i] - wp.vec2(0.0, 0.0)),
        #             gradW(x_i_neighbor, kernel_radius),
        #         )

        delta_density *= dt
        s[i] = ignore_density_diff * (d0 - densities[i]) - delta_density

    @wp.kernel
    def _compute_dii(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        densities: wp.array(dtype=wp.float32),
        kernel_radius: float,
        dt: float,
        dii: wp.array(dtype=wp.vec2),
    ):
        i = wp.tid()
        pos = particles[i]

        temp = wp.vec2(0.0, 0.0)
        query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue

            x_i_neighbor = to2d(pos - particles[query_idx])
            if wp.length(x_i_neighbor) < kernel_radius:
                temp += (
                    1.0
                    / (densities[i] * densities[i] + 1e-7)
                    * gradW(x_i_neighbor, kernel_radius)
                )

        # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
        # query_idx = int(0)
        # while wp.hash_grid_query_next(query, query_idx):
        #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
        #     if wp.length(x_i_neighbor) < kernel_radius:
        #         temp += (
        #             1.0
        #             / (densities[i] * densities[i] + 1e-7)
        #             * gradW(x_i_neighbor, kernel_radius)
        #         )

        temp *= -dt * dt
        dii[i] = temp

    @wp.kernel
    def _compute_aii(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        densities: wp.array(dtype=wp.float32),
        dii: wp.array(dtype=wp.vec2),
        kernel_radius: float,
        dt: float,
        aii: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        pos = particles[i]

        temp = wp.float32(0.0)
        query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue

            x_i_neighbor = to2d(pos - particles[query_idx])
            if wp.length(x_i_neighbor) < kernel_radius:
                dji = (
                    -dt
                    * dt
                    * 1.0
                    / (densities[i] * densities[i] + 1e-7)
                    * gradW(-x_i_neighbor, kernel_radius)
                )

                temp += 1.0 * wp.dot(
                    dii[i] - dji,
                    gradW(x_i_neighbor, kernel_radius),
                )

        # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
        # query_idx = int(0)
        # while wp.hash_grid_query_next(query, query_idx):
        #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
        #     if wp.length(x_i_neighbor) < kernel_radius:
        #         dji = (
        #             -dt
        #             * dt
        #             * 1.0
        #             / (densities[i] * densities[i] + 1e-7)
        #             * gradW(-x_i_neighbor, kernel_radius)
        #         )
        #
        #         temp += 1.0 * wp.dot(
        #             (dii[i] - dji),
        #             gradW(x_i_neighbor, kernel_radius),
        #         )

        aii[i] = temp

    @wp.kernel
    def _update_initial_pressure(pressures: wp.array(dtype=wp.float32), w: float):
        i = wp.tid()
        pressures[i] *= w

    @wp.kernel
    def _compute_sum_dij_pj(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        pressures: wp.array(dtype=wp.float32),
        densities: wp.array(dtype=wp.float32),
        kernel_radius: float,
        d0: float,
        dt: float,
        sum_dij_pj: wp.array(dtype=wp.vec2),
    ):
        i = wp.tid()
        pos = particles[i]

        temp = wp.vec2(0.0, 0.0)
        query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue

            x_i_neighbor = to2d(pos - particles[query_idx])
            if wp.length(x_i_neighbor) < kernel_radius:
                temp += (
                    1.0
                    / (densities[query_idx] * densities[query_idx] + 1e-7)
                    * pressures[query_idx]
                    * gradW(x_i_neighbor, kernel_radius)
                )

        # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
        # query_idx = int(0)
        # while wp.hash_grid_query_next(query, query_idx):
        #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
        #     if wp.length(x_i_neighbor) < kernel_radius:
        #         temp += (
        #             1.0
        #             / (d0 * d0 + 1e-7)
        #             * gradW(x_i_neighbor, kernel_radius)
        #             * pressures[i]  # pressure mirror
        #         )

        temp *= -dt * dt
        sum_dij_pj[i] = temp

    @wp.kernel
    def _compute_sum_dij_pj_boundary(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        pressures: wp.array(dtype=wp.float32),
        densities: wp.array(dtype=wp.float32),
        kernel_radius: float,
        d0: float,
        dt: float,
        sum_dij_pj_boundary: wp.array(dtype=wp.vec2),
    ):
        i = wp.tid()
        pos = boundary_particles[i]

        mirror_pressure = wp.float32(0.0)
        temp = wp.vec2(0.0, 0.0)
        query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            x_i_neighbor = to2d(pos - particles[query_idx])
            if wp.length(x_i_neighbor) < kernel_radius:
                temp += (
                    1.0
                    / (densities[query_idx] * densities[query_idx] + 1e-7)
                    * gradW(x_i_neighbor, kernel_radius)
                    * pressures[query_idx]
                )
                mirror_pressure = pressures[query_idx]

        # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
        # query_idx = int(0)
        # while wp.hash_grid_query_next(query, query_idx):
        #     if query_idx == i:
        #         continue
        #
        #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
        #     if wp.length(x_i_neighbor) < kernel_radius:
        #         temp += (
        #             1.0
        #             / (d0 * d0 + 1e-7)
        #             * gradW(x_i_neighbor, kernel_radius)
        #             * mirror_pressure  # pressure mirror
        #         )

        temp *= -dt * dt
        sum_dij_pj_boundary[i] = temp

    @wp.kernel
    def _compute_new_pressure_err(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        pressures: wp.array(dtype=wp.float32),
        densities: wp.array(dtype=wp.float32),
        kernel_radius: float,
        aii: wp.array(dtype=wp.float32),
        dii: wp.array(dtype=wp.vec2),
        sum_dij_pj: wp.array(dtype=wp.vec2),
        sum_dij_pj_boundary: wp.array(dtype=wp.vec2),
        s: wp.array(dtype=wp.float32),
        w: float,
        d0: float,
        dt: float,
        pressure_tmp: wp.array(dtype=wp.float32),
        err: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        pos = particles[i]

        pressure_tmp[i] = (1.0 - w) * pressures[i]

        temp = wp.float32(0.0)
        query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue

            x_i_neighbor = to2d(pos - particles[query_idx])
            if wp.length(x_i_neighbor) < kernel_radius:
                dji = (
                    -dt
                    * dt
                    * 1.0
                    / (densities[i] * densities[i] + 1e-7)
                    * gradW(-x_i_neighbor, kernel_radius)
                )
                temp += 1.0 * wp.dot(
                    sum_dij_pj[i]
                    - dii[query_idx] * pressures[query_idx]
                    - sum_dij_pj[query_idx]
                    + dji * pressures[i],
                    gradW(x_i_neighbor, kernel_radius),
                )

        # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
        # query_idx = int(0)
        # while wp.hash_grid_query_next(query, query_idx):
        #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
        #     if wp.length(x_i_neighbor) < kernel_radius:
        #         dji = (
        #             -dt
        #             * dt
        #             * 1.0
        #             / (densities[i] * densities[i] + 1e-7)
        #             * gradW(-x_i_neighbor, kernel_radius)
        #         )
        #         temp += 1.0 * wp.dot(
        #             sum_dij_pj[i]
        #             - dii[i] * pressures[i]  # pressure mirror
        #             - sum_dij_pj_boundary[query_idx]
        #             + dji * pressures[i],
        #             gradW(x_i_neighbor, kernel_radius),
        #         )

        pressure_tmp[i] += w / (aii[i] + 1e-7) * (s[i] - temp)
        # pressure_tmp[i] += w / (aii[i]) * (s[i] - temp)
        pressure_tmp[i] = wp.max(pressure_tmp[i], 0.0)
        err[i] = wp.abs(temp + aii[i] * pressures[i] - s[i]) / d0

    @wp.kernel
    def _update_velocities(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
        boundary_particles: wp.array(dtype=wp.vec3),
        boundary_particle_grid: wp.uint64,
        pressures: wp.array(dtype=wp.float32),
        densities: wp.array(dtype=wp.float32),
        kernel_radius: float,
        dt: float,
        velocities: wp.array(dtype=wp.vec2),
    ):
        i = wp.tid()
        pos = particles[i]

        pi = pressures[i] / (densities[i] * densities[i] + 1e-7)

        grad_p = wp.vec2(0.0, 0.0)
        query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
        query_idx = int(0)
        while wp.hash_grid_query_next(query, query_idx):
            if query_idx == i:
                continue

            x_i_neighbor = to2d(pos - particles[query_idx])
            if wp.length(x_i_neighbor) < kernel_radius:
                grad_p += (
                    densities[i]
                    * 1.0
                    * (
                        pi
                        + pressures[query_idx]
                        / (densities[query_idx] * densities[query_idx] + 1e-7)
                    )
                    * gradW(x_i_neighbor, kernel_radius)
                )

        # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
        # query_idx = int(0)
        # while wp.hash_grid_query_next(query, query_idx):
        #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
        #     if wp.length(x_i_neighbor) < kernel_radius:
        #         grad_p += (
        #             densities[i] * 1.0 * 2.0 * pi * gradW(x_i_neighbor, kernel_radius)
        #         )

        velocities[i] += -dt * grad_p / (densities[i] + 1e-7)

    def solve(
        self,
        particles: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec2),  # predicted velocity
        pressures: wp.array(dtype=wp.float32),
        densities: wp.array(dtype=wp.float32),
        dt: float,
    ):
        """
        - all the arguments are after advection
        - velocities are advected + non pressure force
        - if dt == 0.0, then ignore (d0 - density) term
            - (like solve on a grid, we don't consider the time step)
            - assume the density doesn't change according to a div-free velocity field
        """
        ignore_density_diff = 1.0
        if dt == 0.0:
            dt = 1
            ignore_density_diff = 0.0

        wp.launch(
            self._compute_source_term,
            dim=self.n_particles,
            inputs=[
                particles,
                self.particle_grid.id,
                self.boundary_particles,
                self.boundary_particle_grid.id,
                velocities,
                densities,
                self.kernel_radius,
                self.d0,
                dt,
                ignore_density_diff,
                self.s,
            ],
        )
        wp.launch(
            self._compute_dii,
            dim=self.n_particles,
            inputs=[
                particles,
                self.particle_grid.id,
                self.boundary_particles,
                self.boundary_particle_grid.id,
                densities,
                self.kernel_radius,
                dt,
                self.dii,
            ],
        )
        wp.launch(
            self._compute_aii,
            dim=self.n_particles,
            inputs=[
                particles,
                self.particle_grid.id,
                self.boundary_particles,
                self.boundary_particle_grid.id,
                densities,
                self.dii,
                self.kernel_radius,
                dt,
                self.aii,
            ],
        )
        wp.launch(
            self._update_initial_pressure,
            dim=self.n_particles,
            inputs=[pressures, 0.5],  # according to the paper
        )

        iter = 0
        error = 10000
        while iter < self.max_iterations and error > self.error_tolerance:
            wp.launch(
                self._compute_sum_dij_pj,
                dim=self.n_particles,
                inputs=[
                    particles,
                    self.particle_grid.id,
                    self.boundary_particles,
                    self.boundary_particle_grid.id,
                    pressures,
                    densities,
                    self.kernel_radius,
                    self.d0,
                    dt,
                    self.sum_dij_pj,
                ],
            )
            wp.launch(
                self._compute_sum_dij_pj_boundary,
                dim=self.boundary_particles.shape[0],
                inputs=[
                    particles,
                    self.particle_grid.id,
                    self.boundary_particles,
                    self.boundary_particle_grid.id,
                    pressures,
                    densities,
                    self.kernel_radius,
                    self.d0,
                    dt,
                    self.sum_dij_pj_boundary,
                ],
            )
            print("aii: ", self.aii)
            print("dii: ", self.dii)
            print("sum: ", self.sum_dij_pj)
            wp.launch(
                self._compute_new_pressure_err,
                dim=self.n_particles,
                inputs=[
                    particles,
                    self.particle_grid.id,
                    self.boundary_particles,
                    self.boundary_particle_grid.id,
                    pressures,
                    densities,
                    self.kernel_radius,
                    self.aii,
                    self.dii,
                    self.sum_dij_pj,
                    self.sum_dij_pj_boundary,
                    self.s,
                    self.w,
                    self.d0,
                    dt,
                    self.pressure_tmp,
                    self.err,
                ],
            )
            wp.copy(pressures, self.pressure_tmp, count=self.n_particles)
            error = wp.utils.array_sum(self.err) / self.n_particles
            iter += 1
            if iter == 3:
                exit()

        print(f"total iter: {iter}, final error: {error}")
        wp.launch(
            self._update_velocities,
            dim=self.n_particles,
            inputs=[
                particles,
                self.particle_grid.id,
                self.boundary_particles,
                self.boundary_particle_grid.id,
                pressures,
                densities,
                self.kernel_radius,
                dt,
                velocities,
            ],
        )
