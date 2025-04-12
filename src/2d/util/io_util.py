import os
import shutil
import argparse
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from sim.kernel_function import W
from sim.grid import curl


# remove everything in dir
def remove_everything_in(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "y", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")


# p2g, for velocity
@wp.kernel
def p2g(
    particles_3d: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec2),
    grid_velocities: wp.array2d(dtype=wp.vec2),
    hash_grid: wp.uint64,
    kernel_radius: float,
    dx: float,
):
    i, j = wp.tid()
    p = wp.vec3(wp.float32(i), wp.float32(j), 0.0) * dx

    grid_mass = wp.float32(0.0)
    grid_velocity = wp.vec2()

    query = wp.hash_grid_query(hash_grid, p, kernel_radius)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        temp = p - particles_3d[index]
        x_grid_p = wp.vec2(temp.x, temp.y)
        if wp.length(x_grid_p) < kernel_radius:
            grid_mass += W(x_grid_p, kernel_radius) * 1.0
            grid_velocity += velocities[index] * W(x_grid_p, kernel_radius) * 1.0

    # grid_velocities[i, j] = wp.cw_div(grid_velocity, wp.vec2(grid_mass, grid_mass))
    grid_velocities[i, j] = grid_velocity / grid_mass


def to_output_format(
    particles_3d: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    grid_velocities: wp.array(dtype=wp.vec2),
    grid_vorticities: wp.array(dtype=float),
    kernel_radius: float,
    hash_grid: wp.HashGrid,
    dx: float,
):
    wp.launch(
        p2g,
        dim=grid_velocities.shape,
        inputs=[
            particles_3d,
            velocities,
            grid_velocities,
            hash_grid.id,
            kernel_radius,
            dx,
        ],
    )

    wp.launch(
        curl,
        dim=grid_vorticities.shape,
        inputs=[grid_velocities, grid_vorticities, wp.vec2i(grid_velocities.shape), dx],
    )


def dump_data(
    particles_dir: str,
    velocity_dir: str,
    vorticity_dir: str,
    particles: wp.array(dtype=wp.vec2),
    grid_velocities: wp.array(dtype=wp.vec2),
    grid_vorticities: wp.array(dtype=float),
    frame_idx: int,
):
    wp.synchronize()

    pos_cpu = particles.numpy()
    pos_cpu_x, pos_cpu_y = pos_cpu[:, 0], pos_cpu[:, 1]

    vel_cpu = grid_velocities.numpy()
    vel_cpu_x, vel_cpu_y = vel_cpu[:, :, 0].transpose(), vel_cpu[:, :, 1].transpose()

    vort_cpu = grid_vorticities.numpy().transpose()

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()
    ax.set_axis_off()
    plt.scatter(pos_cpu_x, pos_cpu_y, s=3)
    plt.savefig(os.path.join(particles_dir, f"particles_{frame_idx:04d}.png"))
    plt.close(fig)

    figx = 10
    figy = vel_cpu.shape[0] / vel_cpu.shape[1] * figx
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot()
    ax.set_axis_off()
    plt.imshow(vel_cpu_x, cmap="jet", origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(velocity_dir, f"vel_x_{frame_idx:04d}.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot()
    ax.set_axis_off()
    plt.imshow(vel_cpu_y, cmap="jet", origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(velocity_dir, f"vel_y_{frame_idx:04d}.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot()
    ax.set_axis_off()
    plt.imshow(vort_cpu, cmap="jet", origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(vorticity_dir, f"vort_{frame_idx:04d}.png"))
    plt.close(fig)


def dump_boundary_particles(
    particles_dir: str,
    paricles: wp.array(dtype=wp.vec2),
    boundary_particles: wp.array(dtype=wp.vec2),
):
    wp.synchronize()

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot()
    ax.set_axis_off()

    pos_cpu = boundary_particles.numpy()
    pos_cpu_x, pos_cpu_y = pos_cpu[:, 0], pos_cpu[:, 1]
    plt.scatter(pos_cpu_x, pos_cpu_y, s=3, color="green")

    pos_cpu = paricles.numpy()
    pos_cpu_x, pos_cpu_y = pos_cpu[:, 0], pos_cpu[:, 1]
    plt.scatter(pos_cpu_x, pos_cpu_y, s=3, color="blue")

    plt.savefig(os.path.join(particles_dir, "boundary_particles.png"))
    plt.close(fig)
