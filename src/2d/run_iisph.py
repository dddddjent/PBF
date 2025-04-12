import os
import argparse
import datetime
import shutil

import warp as wp
import numpy as np

import util.io_util as io_util
from util.io_util import dump_boundary_particles, remove_everything_in
from util.logger import Logger
from sim.init_conditions import init_leapfrog
from sim.sph import get_initial_density, forward_euler_advection
from sim.poisson import PoissonSolver


#################################### Init #####################################
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Experiment name")
parser.add_argument(
    "--visualize_dt", help="Visualization time step", type=float, default=0.1
)
parser.add_argument("--CFL", help="CFL number", type=float, default=0.5)
parser.add_argument("--from_frame", help="Start frame", type=int, default=0)
parser.add_argument("--total_frames", help="Total frames", type=int, default=4000)
args = parser.parse_args()

visualize_dt = args.visualize_dt
CFL = args.CFL
from_frame = args.from_frame
total_frames = args.total_frames
init_condition = "leapfrog"
init_nx = wp.constant(256)
init_ny = wp.constant(256)
grid_nx = wp.constant(256)
grid_ny = wp.constant(256)
dx = wp.constant(1.0 / init_nx)
kernel_scale = int(3)
kernel_radius = wp.constant(kernel_scale * dx)

exp_name = init_condition + "-iisph"
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

wp.set_device("cuda:0")
wp.config.mode = "debug"
wp.config.verify_cuda = True
wp.config.verify_fp = True
wp.init()
################################################################################

################################## Variables #####################################
particles = wp.zeros(init_nx * init_ny, dtype=wp.vec3)
velocities = wp.zeros(init_nx * init_ny, dtype=wp.vec2)
boundary_particles = wp.zeros(
    (init_nx + 2 * kernel_scale) * (init_ny + 2 * kernel_scale) - init_nx * init_ny,
    dtype=wp.vec3,
)
pressures = wp.zeros(init_nx * init_ny, dtype=wp.float32)
densities = wp.zeros(init_nx * init_ny, dtype=wp.float32)

hash_grid = wp.HashGrid(dim_x=grid_nx, dim_y=grid_ny, dim_z=1)
hash_grid_boundary = wp.HashGrid(
    dim_x=grid_nx + 2 * kernel_scale, dim_y=grid_ny + 2 * kernel_scale, dim_z=1
)

# for visualization
grid_vorticities = wp.zeros(shape=(grid_nx, grid_ny), dtype=float)
grid_velocities = wp.zeros(shape=(grid_nx, grid_ny), dtype=wp.vec2)
################################################################################


def dump_data(frame_idx):
    io_util.to_output_format(
        particles,
        velocities,
        grid_velocities,
        grid_vorticities,
        kernel_radius,
        hash_grid,
        dx,
    )
    io_util.dump_data(
        particles_dir,
        vel_dir,
        vort_dir,
        particles,
        grid_velocities,
        grid_vorticities,
        frame_idx,
    )


def advection(dt):
    wp.launch(
        forward_euler_advection,
        dim=init_nx * init_ny,
        inputs=[particles, velocities, dt],
    )


def main():
    init_leapfrog(
        particles,
        velocities,
        boundary_particles,
        init_nx,
        init_ny,
        dx,
        kernel_scale,
    )
    hash_grid.build(points=particles, radius=kernel_radius)
    hash_grid_boundary.build(points=boundary_particles, radius=kernel_radius)
    d0 = wp.constant(
        get_initial_density(
            particles, hash_grid, boundary_particles, hash_grid_boundary, kernel_radius
        )
    )
    densities = wp.full(value=d0, shape=init_nx * init_ny, dtype=wp.float32)
    solver = PoissonSolver(
        hash_grid,
        boundary_particles,
        hash_grid_boundary,
        init_nx * init_ny,
        kernel_radius,
        d0,
        1e-3,
        100,
    )
    solver.solve(particles, velocities, pressures, densities, 0.0)

    dump_boundary_particles(particles_dir, particles, boundary_particles)
    dump_data(0)

    substeps = 10
    curr_dt = visualize_dt / substeps
    for frame in range(from_frame + 1, total_frames + 1):
        for _ in range(substeps):
            advection(curr_dt)
            hash_grid.build(points=particles, radius=kernel_radius)
            solver.solve(particles, velocities, pressures, densities, curr_dt)
        print(frame)
        dump_data(frame)


if __name__ == "__main__":
    main()
    wp.synchronize()
