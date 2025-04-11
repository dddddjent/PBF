import os
import argparse
import datetime
import shutil

import warp as wp

from util.io_util import remove_everything_in
from util.logger import Logger
from sim.init_conditions import init_leapfrog


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

wp.set_device("cuda:0")
wp.config.mode = "debug"
wp.config.verify_cuda = True
wp.config.verify_fp = True
wp.init()
################################################################################

################################## Variables #####################################
particles = wp.zeros(init_nx * init_ny, dtype=wp.vec2)
velocities = wp.zeros(init_nx * init_ny, dtype=wp.vec2)
boundary_particles = wp.zeros(
    (init_nx + 2 * kernel_scale) * (init_ny + 2 * kernel_scale) - init_nx * init_ny,
    dtype=wp.vec2,
)
################################################################################


def main():
    vortdir = "vort"
    vortdir = os.path.join(logsdir, vortdir)
    os.makedirs(vortdir, exist_ok=True)
    veldir = "vel"
    veldir = os.path.join(logsdir, veldir)
    os.makedirs(veldir, exist_ok=True)
    particlesdir = "particles"
    particlesdir = os.path.join(logsdir, particlesdir)
    os.makedirs(particlesdir, exist_ok=True)
    shutil.copy(__file__, logsdir)

    init_leapfrog(
        particles,
        velocities,
        boundary_particles,
        init_nx,
        init_ny,
        dx,
        kernel_scale,
        kernel_radius,
    )


if __name__ == "__main__":
    main()
