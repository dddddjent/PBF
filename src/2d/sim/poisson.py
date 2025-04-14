import warp as wp
from sim.kernel_function import W, gradW
from util.warp_util import to2d, to3d
from util.io_util import debug_particle_field, debug_particle_vector_field
import sim.sph as sph


# # Static boundary conditions
# class IISPHPoissonSolver:
#     def __init__(
#         self,
#         particle_grid: wp.HashGrid,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.HashGrid,
#         n_particles: int,
#         kernel_radius: float,
#         d0: float,  # initial density
#         error_tolerance: float = 1e-4,
#         max_iterations: int = 100,
#     ):
#         self.particle_grid = particle_grid
#         self.boundary_particles = boundary_particles
#         self.boundary_particle_grid = boundary_particle_grid
#         self.n_particles = n_particles
#         self.kernel_radius = kernel_radius
#         self.d0 = d0
#         self.error_tolerance = error_tolerance
#         self.max_iterations = max_iterations
#
#         self.aii = wp.zeros(n_particles, dtype=wp.float32)  # a_ii in A
#         self.dii = wp.zeros(n_particles, dtype=wp.vec2)
#         self.sum_dij_pj = wp.zeros(n_particles, dtype=wp.vec2)
#         self.sum_dij_pj_boundary = wp.zeros(boundary_particles.shape[0], dtype=wp.vec2)
#         self.s = wp.zeros(n_particles, dtype=wp.float32)
#         self.w = wp.constant(wp.float32(0.5))
#         self.pressure_tmp = wp.zeros(n_particles, dtype=wp.float32)
#         self.err = wp.zeros(n_particles, dtype=wp.float32)
#
#     @wp.kernel
#     def _compute_source_term(
#         particles: wp.array(dtype=wp.vec3),
#         particle_grid: wp.uint64,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.uint64,
#         velocities: wp.array(dtype=wp.vec2),
#         densities: wp.array(dtype=wp.float32),
#         kernel_radius: float,
#         d0: float,
#         dt: float,
#         ignore_density_diff: float,
#         s: wp.array(dtype=wp.float32),
#     ):
#         i = wp.tid()
#         pos = particles[i]
#
#         delta_density = float(0.0)
#         query = wp.hash_grid_query(particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             if query_idx == i:
#                 continue
#
#             x_i_neighbor = to2d(pos - particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 delta_density += 1.0 * wp.dot(
#                     (velocities[i] - velocities[query_idx]),
#                     gradW(x_i_neighbor, kernel_radius),
#                 )
#
#         query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             x_i_neighbor = to2d(pos - boundary_particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 delta_density += 1.0 * wp.dot(
#                     (velocities[i] - wp.vec2(0.0, 0.0)),
#                     gradW(x_i_neighbor, kernel_radius),
#                 )
#
#         delta_density *= dt
#         s[i] = ignore_density_diff * (d0 - densities[i]) - delta_density
#
#     @wp.kernel
#     def _compute_dii(
#         particles: wp.array(dtype=wp.vec3),
#         particle_grid: wp.uint64,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.uint64,
#         densities: wp.array(dtype=wp.float32),
#         kernel_radius: float,
#         dt: float,
#         dii: wp.array(dtype=wp.vec2),
#     ):
#         i = wp.tid()
#         pos = particles[i]
#
#         temp = wp.vec2(0.0, 0.0)
#         query = wp.hash_grid_query(particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             if query_idx == i:
#                 continue
#
#             x_i_neighbor = to2d(pos - particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 temp += (
#                     1.0
#                     / (densities[i] * densities[i])
#                     * gradW(x_i_neighbor, kernel_radius)
#                 )
#
#         query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             x_i_neighbor = to2d(pos - boundary_particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 temp += (
#                     1.0
#                     / (densities[i] * densities[i] + 1e-7)
#                     * gradW(x_i_neighbor, kernel_radius)
#                 )
#
#         temp *= -dt * dt
#         dii[i] = temp
#
#     @wp.kernel
#     def _compute_aii(
#         particles: wp.array(dtype=wp.vec3),
#         particle_grid: wp.uint64,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.uint64,
#         densities: wp.array(dtype=wp.float32),
#         dii: wp.array(dtype=wp.vec2),
#         kernel_radius: float,
#         dt: float,
#         aii: wp.array(dtype=wp.float32),
#     ):
#         i = wp.tid()
#         pos = particles[i]
#
#         temp = wp.float32(0.0)
#         query = wp.hash_grid_query(particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             if query_idx == i:
#                 continue
#
#             x_i_neighbor = to2d(pos - particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 dji = (
#                     -dt
#                     * dt
#                     * 1.0
#                     / (densities[i] * densities[i])
#                     * gradW(-x_i_neighbor, kernel_radius)
#                 )
#
#                 temp += 1.0 * wp.dot(
#                     dii[i] - dji,
#                     gradW(x_i_neighbor, kernel_radius),
#                 )
#
#         query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             x_i_neighbor = to2d(pos - boundary_particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 dji = (
#                     -dt
#                     * dt
#                     * 1.0
#                     / (densities[i] * densities[i] + 1e-7)
#                     * gradW(-x_i_neighbor, kernel_radius)
#                 )
#
#                 temp += 1.0 * wp.dot(
#                     dii[i] - dji,
#                     gradW(x_i_neighbor, kernel_radius),
#                 )
#
#         aii[i] = temp
#
#     @wp.kernel
#     def _update_initial_pressure(pressures: wp.array(dtype=wp.float32), w: float):
#         i = wp.tid()
#         pressures[i] *= w
#
#     @wp.kernel
#     def _compute_sum_dij_pj(
#         particles: wp.array(dtype=wp.vec3),
#         particle_grid: wp.uint64,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.uint64,
#         pressures: wp.array(dtype=wp.float32),
#         densities: wp.array(dtype=wp.float32),
#         kernel_radius: float,
#         d0: float,
#         dt: float,
#         sum_dij_pj: wp.array(dtype=wp.vec2),
#     ):
#         i = wp.tid()
#         pos = particles[i]
#
#         temp = wp.vec2(0.0, 0.0)
#         query = wp.hash_grid_query(particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             if query_idx == i:
#                 continue
#
#             x_i_neighbor = to2d(pos - particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 temp += (
#                     1.0
#                     / (densities[query_idx] * densities[query_idx])
#                     * pressures[query_idx]
#                     * gradW(x_i_neighbor, kernel_radius)
#                 )
#
#         # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
#         # query_idx = int(0)
#         # while wp.hash_grid_query_next(query, query_idx):
#         #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
#         #     if wp.length(x_i_neighbor) < kernel_radius:
#         #         temp += (
#         #             1.0
#         #             / (d0 * d0 + 1e-7)
#         #             * gradW(x_i_neighbor, kernel_radius)
#         #             * pressures[i]  # pressure mirror
#         #         )
#
#         temp *= -dt * dt
#         sum_dij_pj[i] = temp
#
#     @wp.kernel
#     def _compute_sum_dij_pj_boundary(
#         particles: wp.array(dtype=wp.vec3),
#         particle_grid: wp.uint64,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.uint64,
#         pressures: wp.array(dtype=wp.float32),
#         densities: wp.array(dtype=wp.float32),
#         kernel_radius: float,
#         d0: float,
#         dt: float,
#         sum_dij_pj_boundary: wp.array(dtype=wp.vec2),
#     ):
#         i = wp.tid()
#         pos = boundary_particles[i]
#
#         mirror_pressure = wp.float32(0.0)
#         temp = wp.vec2(0.0, 0.0)
#         query = wp.hash_grid_query(particle_grid, pos, kernel_radius)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             x_i_neighbor = to2d(pos - particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius:
#                 temp += (
#                     1.0
#                     / (densities[query_idx] * densities[query_idx] + 1e-7)
#                     * gradW(x_i_neighbor, kernel_radius)
#                     * pressures[query_idx]
#                 )
#                 mirror_pressure = pressures[query_idx]
#
#         # query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius)
#         # query_idx = int(0)
#         # while wp.hash_grid_query_next(query, query_idx):
#         #     if query_idx == i:
#         #         continue
#         #
#         #     x_i_neighbor = to2d(pos - boundary_particles[query_idx])
#         #     if wp.length(x_i_neighbor) < kernel_radius:
#         #         temp += (
#         #             1.0
#         #             / (d0 * d0 + 1e-7)
#         #             * gradW(x_i_neighbor, kernel_radius)
#         #             * mirror_pressure  # pressure mirror
#         #         )
#
#         temp *= -dt * dt
#         sum_dij_pj_boundary[i] = temp
#
#     @wp.kernel
#     def _compute_new_pressure_err(
#         particles: wp.array(dtype=wp.vec3),
#         particle_grid: wp.uint64,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.uint64,
#         pressures: wp.array(dtype=wp.float32),
#         densities: wp.array(dtype=wp.float32),
#         kernel_radius: float,
#         aii: wp.array(dtype=wp.float32),
#         dii: wp.array(dtype=wp.vec2),
#         sum_dij_pj: wp.array(dtype=wp.vec2),
#         sum_dij_pj_boundary: wp.array(dtype=wp.vec2),
#         s: wp.array(dtype=wp.float32),
#         w: float,
#         d0: float,
#         dt: float,
#         pressure_tmp: wp.array(dtype=wp.float32),
#         err: wp.array(dtype=wp.float32),
#         accelerations: wp.array(dtype=wp.vec2),
#         debug_field: wp.array(dtype=wp.float32),
#     ):
#         i = wp.tid()
#         pos = particles[i]
#
#         accelerations[i] = wp.vec2(0.0, 0.0)
#         accelerations[i] = (dii[i] * pressures[i] + sum_dij_pj[i]) / (dt * dt)
#         Ap_i = wp.float32(0.0)
#         query = wp.hash_grid_query(particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             if query_idx == i:
#                 continue
#
#             x_i_neighbor = to2d(pos - particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 Ap_i += 1.0 * wp.dot(
#                     dii[i] * pressures[i]
#                     + sum_dij_pj[i]
#                     - dii[query_idx] * pressures[query_idx]
#                     - sum_dij_pj[query_idx],
#                     gradW(x_i_neighbor, kernel_radius),
#                 )
#
#         query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             x_i_neighbor = to2d(pos - boundary_particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 Ap_i += 1.0 * wp.dot(
#                     sum_dij_pj[i] + dii[i] * pressures[i],
#                     gradW(x_i_neighbor, kernel_radius),
#                 )
#
#         # debug_field[i] = w / aii[i] * (s[i])
#         # debug_field[i] = float(cnt)
#         # debug_field[i] = Ap_i
#         pressure_tmp[i] = pressures[i] + w / aii[i] * (s[i] - Ap_i)
#         pressure_tmp[i] = wp.max(pressure_tmp[i], 0.0)
#         err[i] = wp.abs(Ap_i - s[i]) / d0
#
#     @wp.kernel
#     def _update_velocities(
#         particles: wp.array(dtype=wp.vec3),
#         particle_grid: wp.uint64,
#         boundary_particles: wp.array(dtype=wp.vec3),
#         boundary_particle_grid: wp.uint64,
#         pressures: wp.array(dtype=wp.float32),
#         densities: wp.array(dtype=wp.float32),
#         kernel_radius: float,
#         d0: float,
#         dt: float,
#         accelerations: wp.array(dtype=wp.vec2),
#         velocities: wp.array(dtype=wp.vec2),
#     ):
#         i = wp.tid()
#         pos = particles[i]
#
#         pi = pressures[i] / (densities[i] * densities[i])
#
#         acc = wp.vec2(0.0, 0.0)
#         query = wp.hash_grid_query(particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             if query_idx == i:
#                 continue
#
#             x_i_neighbor = to2d(pos - particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 acc += (
#                     1.0
#                     * (
#                         pi
#                         + pressures[query_idx]
#                         / (densities[query_idx] * densities[query_idx])
#                     )
#                     * gradW(x_i_neighbor, kernel_radius)
#                 )
#
#         query = wp.hash_grid_query(boundary_particle_grid, pos, kernel_radius * 2.0)
#         query_idx = int(0)
#         while wp.hash_grid_query_next(query, query_idx):
#             x_i_neighbor = to2d(pos - boundary_particles[query_idx])
#             if wp.length(x_i_neighbor) < kernel_radius * 2.0:
#                 acc += (
#                     1.0
#                     # * (pi + pressures[i] / (d0 * d0))
#                     * pi
#                     * gradW(x_i_neighbor, kernel_radius)
#                 )
#
#         accelerations[i] = -acc
#         velocities[i] += -dt * acc
#
#     def solve(
#         self,
#         particles: wp.array(dtype=wp.vec3),
#         velocities: wp.array(dtype=wp.vec2),  # predicted velocity
#         pressures: wp.array(dtype=wp.float32),
#         densities: wp.array(dtype=wp.float32),
#         dt: float,
#         frame: int,
#     ):
#         """
#         - all the arguments are after advection
#         - velocities are advected + non pressure force
#         - if dt == 0.0, then ignore (d0 - density) term
#             - (like solve on a grid, we don't consider the time step)
#             - assume the density doesn't change according to a div-free velocity field
#         """
#         debug_field = wp.zeros(self.n_particles, dtype=wp.float32)
#         ignore_density_diff = 1.0
#         if dt == 0.0:
#             dt = 1
#             ignore_density_diff = 0.0
#
#         wp.launch(
#             self._compute_source_term,
#             dim=self.n_particles,
#             inputs=[
#                 particles,
#                 self.particle_grid.id,
#                 self.boundary_particles,
#                 self.boundary_particle_grid.id,
#                 velocities,
#                 densities,
#                 self.kernel_radius,
#                 self.d0,
#                 dt,
#                 ignore_density_diff,
#                 self.s,
#             ],
#         )
#         wp.launch(
#             self._compute_dii,
#             dim=self.n_particles,
#             inputs=[
#                 particles,
#                 self.particle_grid.id,
#                 self.boundary_particles,
#                 self.boundary_particle_grid.id,
#                 densities,
#                 self.kernel_radius,
#                 dt,
#                 self.dii,
#             ],
#         )
#         wp.launch(
#             self._compute_aii,
#             dim=self.n_particles,
#             inputs=[
#                 particles,
#                 self.particle_grid.id,
#                 self.boundary_particles,
#                 self.boundary_particle_grid.id,
#                 densities,
#                 self.dii,
#                 self.kernel_radius,
#                 dt,
#                 self.aii,
#             ],
#         )
#         wp.launch(
#             self._update_initial_pressure,
#             dim=self.n_particles,
#             inputs=[pressures, 0.5],  # according to the paper
#         )
#
#         acc = wp.zeros(shape=self.n_particles, dtype=wp.vec2)
#         iter = 0
#         error = 10000
#         wp.synchronize()
#         # print((self.s.numpy() / self.aii.numpy())[128 * 64 : 128 * 64 + 20])
#         # print(self.s.numpy()[128 * 64 : 128 * 64 + 20])
#         # print(self.aii.numpy()[128 * 64 : 128 * 64 + 20])
#         # print(velocities.numpy()[128 * 64 : 128 * 64 + 20])
#         # print(self.dii.numpy()[128 * 64 : 128 * 64 + 20])
#         # exit()
#         while iter < self.max_iterations and error > self.error_tolerance:
#             wp.launch(
#                 self._compute_sum_dij_pj,
#                 dim=self.n_particles,
#                 inputs=[
#                     particles,
#                     self.particle_grid.id,
#                     self.boundary_particles,
#                     self.boundary_particle_grid.id,
#                     pressures,
#                     densities,
#                     self.kernel_radius,
#                     self.d0,
#                     dt,
#                     self.sum_dij_pj,
#                 ],
#             )
#             wp.synchronize()
#             # print(self.sum_dij_pj.numpy()[128 * 64 : 128 * 64 + 10])
#             wp.launch(
#                 self._compute_new_pressure_err,
#                 dim=self.n_particles,
#                 inputs=[
#                     particles,
#                     self.particle_grid.id,
#                     self.boundary_particles,
#                     self.boundary_particle_grid.id,
#                     pressures,
#                     densities,
#                     self.kernel_radius,
#                     self.aii,
#                     self.dii,
#                     self.sum_dij_pj,
#                     self.sum_dij_pj_boundary,
#                     self.s,
#                     self.w,
#                     self.d0,
#                     dt,
#                     self.pressure_tmp,
#                     self.err,
#                     acc,
#                     debug_field,
#                 ],
#             )
#             wp.copy(pressures, self.pressure_tmp, count=self.n_particles)
#             # print(pressures.numpy()[128 * 64 : 128 * 64 + 20])
#             # print(debug_field.numpy()[128 * 64 : 128 * 64 + 20])
#             error = wp.utils.array_sum(self.err) / self.n_particles
#             iter += 1
#             if iter % 10 == 0:
#                 print(f"iter: {iter}, error: {error}")
#             # print(error)
#
#         # print(pressures.numpy()[128 * 64 : 128 * 64 + 20])
#         # debug_particle_field("./", particles, pressures, f"pressure{frame}")
#         # debug_particle_field("./", particles, debug_field, f"debug{frame}")
#         # debug_particle_field("./", particles, self.aii, f"aii{frame}")
#         # debug_particle_vector_field("./", particles, self.dii, f"dii{frame}")
#         # print("dii: ", self.dii)
#         # print("aii: ", self.aii)
#         print(f"total iter: {iter}, final error: {error}")
#         # debug_particle_vector_field(
#         #     "./", particles, acc, f"accelerations{frame}-1", normalize=True
#         # )
#         # print(acc.numpy()[128 * 64 : 128 * 64 + 20])
#         wp.launch(
#             self._update_velocities,
#             dim=self.n_particles,
#             inputs=[
#                 particles,
#                 self.particle_grid.id,
#                 self.boundary_particles,
#                 self.boundary_particle_grid.id,
#                 pressures,
#                 densities,
#                 self.kernel_radius,
#                 self.d0,
#                 dt,
#                 acc,
#                 velocities,
#             ],
#         )
#         debug_particle_vector_field(
#             "./", particles, acc, f"accelerations{frame}", normalize=True
#         )
#         # print(acc.numpy()[128 * 64 : 128 * 64 + 20])
#         # debug_particle_vector_field(
#         #     "./", particles, velocities, f"velocities{frame}", normalize=False
#         # )


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
        gradC[i] /= d0

    @wp.kernel
    def compute_lambdas(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
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

        lambdas[i] = lambdas[i] + (-C[i] - alphas[i] / (dt * dt) * lambdas[i]) / (
            alphas[i] / (dt * dt) + temp
        )

    @wp.kernel
    def update_positions(
        particles: wp.array(dtype=wp.vec3),
        particle_grid: wp.uint64,
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
