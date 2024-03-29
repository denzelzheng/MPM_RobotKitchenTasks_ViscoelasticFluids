import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim import RigidBody, MpmLagSim, MpmTetLagSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh, sdf_from_mesh, transform_pos


def test_sdf_from_mesh(mesh_pth='./data/cut/cut0001/knife.obj', res=128):
    mesh = trimesh.load_mesh(mesh_pth)
    sdf = sdf_from_mesh(mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    inds = np.where(np.abs(sdf.np_voxels.reshape((-1, 1))) < 1e-3)[0]
    colors = np.array([[0.2, 0.3, 0.7, 1]]).repeat(inds.shape[0], axis=0)
    x = y = z = np.arange(res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    sdf_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))[inds]
    
    # sdf_points_color = np.zeros((colors.shape[0], 4), dtype=np.uint8)
    # sdf_points_color[:, -1] = np.uint8(0.3 * 255)
    # sdf_points_color[:, :3] = (colors * np.array([1, 1, 1])).astype(np.uint8)
    print(inds.shape, sdf_points.shape, colors.shape)
    sdf_pcd = trimesh.points.PointCloud(sdf_points, colors=colors)

    display_vertices = transform_pos(mesh.vertices, sdf.np_T_mesh2vox)

    display_mesh = trimesh.Trimesh(vertices=display_vertices,
                                            faces=mesh.faces)

    # meshes = [display_mesh, sdf_pcd]
    meshes = [sdf_pcd]
    trimesh_show(meshes)


if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    test_sdf_from_mesh()