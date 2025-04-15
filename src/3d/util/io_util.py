import os
import shutil
import argparse
import pyvista as pv
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from sim.kernel_function import W
from util.warp_util import to2d, to3d


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


def dump_boundary_particles(
    dir: str,
    particles: wp.array,
    boundary_particles: wp.array,
    append_name: str = "",
    extra_properties: dict[str, wp.array] = None,
):
    wp.synchronize()

    grid = pv.UnstructuredGrid()
    grid.points = particles.numpy()
    if extra_properties is not None:
        for k, v in extra_properties.items():
            grid.point_data[k] = v.numpy()
    grid.save(os.path.join(dir, f"particles{append_name}.vtk"))

    grid = pv.UnstructuredGrid()
    grid.points = boundary_particles.numpy()
    grid.save(os.path.join(dir, f"boundary_particles{append_name}.vtk"))
