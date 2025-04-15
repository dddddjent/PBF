from numpy.matlib import interp
import trimesh
import skimage
import numpy as np
import pyvista as pv
import warp as wp
import math
from grid import interp_2
import skfmm


def create_trs_matrix(translation, rotation, scale):
    """
    Create a 4x4 transformation matrix from translation, rotation (Euler angles), and scale.

    Parameters:
    - translation: (tx, ty, tz)
    - rotation: (rx, ry, rz) in radians
    - scale: (sx, sy, sz)

    Returns:
    - 4x4 NumPy array representing the transformation matrix.
    """
    tx, ty, tz = translation
    rx, ry, rz = rotation
    sx, sy, sz = scale

    # Compute rotation matrices around x, y, z axes
    cx, sx_sin = np.cos(rx), np.sin(rx)
    cy, sy_sin = np.cos(ry), np.sin(ry)
    cz, sz_sin = np.cos(rz), np.sin(rz)

    # Rotation matrix around X-axis
    Rx = np.array([[1, 0, 0], [0, cx, -sx_sin], [0, sx_sin, cx]])

    # Rotation matrix around Y-axis
    Ry = np.array([[cy, 0, sy_sin], [0, 1, 0], [-sy_sin, 0, cy]])

    # Rotation matrix around Z-axis
    Rz = np.array([[cz, -sz_sin, 0], [sz_sin, cz, 0], [0, 0, 1]])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    # Apply scaling
    R_scaled = R * np.array([sx, sy, sz])

    # Construct the 4x4 transformation matrix
    trs_matrix = np.eye(4)
    trs_matrix[:3, :3] = R_scaled
    trs_matrix[:3, 3] = [tx, ty, tz]

    return trs_matrix


# suppose in put and output spacings are all 1.0
@wp.kernel
def transform_voxels(
    voxels: wp.array3d(dtype=wp.float32),
    transformed_voxels: wp.array3d(dtype=wp.float32),
    mat: wp.mat44,
):
    i, j, k = wp.tid()  # in output voxels
    transformed_pos = wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))
    p4 = mat @ wp.vec4(transformed_pos[0], transformed_pos[1], transformed_pos[2], 1.0)
    p = wp.vec3(p4[0], p4[1], p4[2])

    dim = wp.vec3i(voxels.shape[0], voxels.shape[1], voxels.shape[2])
    value = interp_2(voxels, p, dim)

    transformed_voxels[i, j, k] = value


voxels = np.load("obj/bunny_raw.npy")
voxels = wp.array(voxels.astype(np.float32), dtype=wp.float32)

output_shape = (128, 128, 128)
transformed_voxels = np.zeros(shape=output_shape, dtype=np.float32)
transformed_voxels = wp.array(transformed_voxels, dtype=wp.float32)

mat = create_trs_matrix(
    (70, 110, 10), (math.pi / 2, 0, math.pi / 6), (0.35, 0.35, 0.35)
)
mat = np.linalg.inv(mat).astype(np.float32)
mat = wp.mat44(mat)

wp.launch(
    transform_voxels,
    dim=output_shape,
    inputs=[voxels, transformed_voxels, mat],
)
wp.synchronize()

transformed_voxels = transformed_voxels.numpy()
transformed_voxels = skfmm.distance(transformed_voxels, dx=1.0)
np.save("obj/bunny_transformed.npy", transformed_voxels)

nx, ny, nz = transformed_voxels.shape
spacing = (1.0, 1.0, 1.0)
origin = (0.0, 0.0, 0.0)
grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=spacing, origin=origin)
grid.cell_data["SDF"] = transformed_voxels.flatten(order="F")
grid.save("obj/sdf_output.vti")
