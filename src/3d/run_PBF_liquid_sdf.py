import os
import argparse
import datetime
import shutil

import warp as wp

wp.set_device("cuda:0")
wp.config.mode = "release"
# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# wp.config.verify_fp = True
wp.config.cache_kernels = False
wp.init()

import numpy as np

import sim.sph as sph
import util.io_util as io_util
from util.timer import Timer
from util.io_util import (
    dump_boundary_particles,
    remove_everything_in,
)
from util.logger import Logger
from sim.init_conditions import init_liquid
from sim.sph import (
    get_initial_density,
    forward_euler_advection,
    apply_acc,
    damp,
    apply_viscosity,
    compute_vorticity,
    compute_vorticity_confinement_acc,
)
from sim.poisson import PBF_SDF_PossionSolver

#################################### Init #####################################
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Experiment name")
parser.add_argument(
    "--visualize_dt", help="Visualization time step", type=float, default=0.1
)
parser.add_argument("--CFL", help="CFL number", type=float, default=0.5)
parser.add_argument("--from_frame", help="Start frame", type=int, default=0)
parser.add_argument("--total_frames", help="Total frames", type=int, default=450)
parser.add_argument(
    "--damping_speed",
    help="Percentage of velocity remains after 1 second",
    type=float,
    default=0.01,
)
args = parser.parse_args()

visualize_dt = args.visualize_dt
CFL = args.CFL
from_frame = args.from_frame
total_frames = args.total_frames
init_condition = "liquid"
init_nx = 80
init_ny = 80
init_nz = 100
n_particles = init_nx * init_ny * init_nz
grid_nx = wp.constant(128)
grid_ny = wp.constant(128)
grid_nz = wp.constant(128)
dx = wp.constant(1.0)
kernel_scale = int(3)
kernel_radius = wp.constant(3.0 * dx)
damping_speed = args.damping_speed

exp_name = init_condition + "-PBF"
if args.name is None:
    exp_name += datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
else:
    exp_name = args.name

logsdir = os.path.join("logs", exp_name)
os.makedirs(logsdir, exist_ok=True)

from_frame = max(0, from_frame)
if from_frame <= 0 and args.name is not None:
    remove_everything_in(logsdir)

logger = Logger(os.path.join(logsdir, "log.txt"))

vort_dir = "vort"
vort_dir = os.path.join(logsdir, vort_dir)
os.makedirs(vort_dir, exist_ok=True)
vel_dir = "vel"
vel_dir = os.path.join(logsdir, vel_dir)
os.makedirs(vel_dir, exist_ok=True)
particles_dir = "particles"
particles_dir = os.path.join(logsdir, particles_dir)
os.makedirs(particles_dir, exist_ok=True)
shutil.copy(__file__, logsdir)

timer = Timer()

################################################################################

################################## Variables #####################################
particles = wp.zeros(n_particles, dtype=wp.vec3)
particles_pred = wp.zeros(n_particles, dtype=wp.vec3)
velocities = wp.zeros(n_particles, dtype=wp.vec3)
densities = wp.zeros(n_particles, dtype=wp.float32)
alphas = wp.full(n_particles, 1e-3, dtype=wp.float32)
nb_particles = 0
boundary_particles = wp.zeros(nb_particles, dtype=wp.vec3)
velocities_temp = wp.zeros(n_particles, dtype=wp.vec3)

acc_external = wp.zeros(n_particles, dtype=wp.vec3)
vorticities = wp.zeros(n_particles, dtype=wp.vec3)

SDF = np.load("obj/bunny_transformed.npy")
SDF = wp.array(SDF, dtype=wp.float32)

particle_grid = wp.HashGrid(dim_x=grid_nx, dim_y=grid_ny, dim_z=grid_nz)
boundary_grid = wp.HashGrid(
    dim_x=grid_nx + 2 * kernel_scale,
    dim_y=grid_ny + 2 * kernel_scale,
    dim_z=grid_nz + 2 * kernel_scale,
)
################################################################################


@wp.kernel
def update_velocity(
    particles: wp.array(dtype=wp.vec3),
    particles_pred: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    p = particles_pred[i]
    v = p - particles[i]
    v = v / dt
    velocities[i] = v


def advection_predict(dt):
    wp.launch(
        forward_euler_advection,
        dim=n_particles,
        inputs=[particles, velocities, dt, particles_pred],
    )


def update_density():
    wp.launch(
        sph.update_density,
        dim=n_particles,
        inputs=[
            particles,
            particle_grid.id,
            boundary_particles,
            boundary_grid.id,
            kernel_radius,
            densities,
        ],
    )


def vorticity_confinement(strength=0.1):
    wp.launch(
        sph.compute_vorticity,
        dim=n_particles,
        inputs=[
            particles,
            particle_grid.id,
            velocities,
            kernel_radius,
            vorticities,
        ],
    )
    wp.launch(
        sph.compute_vorticity_confinement_acc,
        dim=n_particles,
        inputs=[
            particles,
            particle_grid.id,
            vorticities,
            densities,
            kernel_radius,
            acc_external,
            strength,
        ],
    )


def apply_external_force(dt):
    acc_external.zero_()

    vorticity_confinement(0.1)
    wp.launch(
        apply_acc,
        dim=n_particles,
        inputs=[velocities, acc_external, dt],
    )


def main():
    global boundary_particles, nb_particles
    boundary_particles, nb_particles = init_liquid(
        particles,
        velocities,
        init_nx,
        init_ny,
        init_nz,
        grid_nx,
        grid_ny,
        grid_nz,
        n_particles,
        dx,
        kernel_scale,
    )
    particle_grid.build(points=particles, radius=kernel_radius * 1.2)
    boundary_grid.build(points=boundary_particles, radius=kernel_radius * 1.2)
    d0 = wp.constant(
        get_initial_density(
            particles,
            particle_grid,
            boundary_particles,
            boundary_grid,
            kernel_radius,
        )
    )
    logger.info(f"d0: {float(d0)}")
    update_density()
    solver = PBF_SDF_PossionSolver(
        particle_grid,
        boundary_particles,
        boundary_grid,
        alphas,
        SDF,
        n_particles,
        kernel_radius,
        d0,
        1e-2,
        400,
        (grid_nx, grid_ny, grid_nz),
        k_corr=-0.01,
    )

    dump_boundary_particles(particles_dir, particles, boundary_particles)

    substeps = 3
    curr_dt = visualize_dt / substeps
    damping_factor = (1.0 - damping_speed) ** (curr_dt / 1.0)
    for frame in range(from_frame + 1, total_frames + 1):
        timer.reset()
        for substep_idx in range(substeps):
            apply_external_force(curr_dt)
            advection_predict(curr_dt)
            solver.solve(particles_pred, densities, curr_dt, f"{frame}_{substep_idx}")
            wp.launch(
                update_velocity,
                dim=n_particles,
                inputs=[particles, particles_pred, velocities, curr_dt],
            )
            particle_grid.build(points=particles_pred, radius=kernel_radius)
            wp.launch(
                apply_viscosity,
                dim=n_particles,
                inputs=[
                    particles,
                    particle_grid.id,
                    velocities,
                    kernel_radius,
                    velocities_temp,
                    0.01,
                ],
            )
            wp.launch(
                damp,
                dim=n_particles,
                inputs=[velocities_temp, damping_factor],
            )
            wp.copy(velocities, velocities_temp, count=n_particles)
            wp.copy(particles, particles_pred, count=n_particles)

        print(f"frame: {frame}, {timer.elapsed()}")

        dump_boundary_particles(
            particles_dir,
            particles,
            boundary_particles,
            f"{frame}",
            {"density": densities},
        )


if __name__ == "__main__":
    main()
    wp.synchronize()
