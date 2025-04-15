# call from src/3d

from mesh_to_sdf import mesh_to_voxels
import numpy as np

import trimesh

mesh = trimesh.load("obj/stanford-bunny.obj")
# mesh.apply_scale([0.1, 0.1, 0.1])

voxels = -mesh_to_voxels(
    # voxels = mesh_to_voxels(
    mesh,
    voxel_resolution=128 - 2,
    surface_point_method="sample",
    pad=True,
    sign_method="normal",
)
print(voxels.shape)

np.save("obj/bunny_raw.npy", voxels)

# vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()
