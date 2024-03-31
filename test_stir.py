import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim import StaticBoundary, DynamicBoundary, RigidBody, SoftBody, NeoHookean, MpmSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh


def test_sim():
    ti.init(arch=ti.cuda)
    dt = 1e-4
    substeps = 5

    cut_folder = './data/cut/cut0001/'

    nhk = NeoHookean(1e5, 0)
    dumpling_mesh = trimesh.load_mesh(pjoin(cut_folder, 'dumpling1.obj'))
    dumpling_points = dumpling_mesh.sample(8192)
    dumpling_points += np.array([0, -0.1, 0])
    dumpling = SoftBody(dumpling_points, nhk)

    chopping_board_mesh = trimesh.load_mesh(pjoin(cut_folder, 'chopping_board.obj'))
    chopping_board_mesh.vertices += np.array([0.5, 0.4, 0.5])
    chopping_board = StaticBoundary(mesh=chopping_board_mesh)
    
    knife_mesh = trimesh.load_mesh(pjoin(cut_folder, 'knife.obj'))
    for m in knife_mesh.geometry.values():
        m.vertices += np.array([0.5, 0.7, 0.5])
    knife = DynamicBoundary(mesh=knife_mesh, collide_type="both")
    # knife = StaticBoundary(mesh=knife_mesh)


    sim = MpmSim(origin=np.asarray([-0.5,] * 3), dt=dt, ground_friction=0, box_bound_rel=0.1)
    sim.set_camera_pos(0.5, 0.6, 1.5)
    sim.camera_lookat(0.5, 0.5, 0.5)
    # sim.set_camera_pos(0.75, 1, 0.3)
    sim.add_boundary(chopping_board)
    sim.add_boundary(knife)
    sim.add_body(dumpling)

    # sim.add_lag_body(dumpling_mesh, 1000, 0.1)

    sim.init_system()

    print("start simulation...")
    print("({} static and {} dynamic boundary)".format(sim.n_static_bounds, sim.n_dynamic_bounds))
    frame = 0
    x, y, z = 0, 0, 0
    down = True
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        for s in range(substeps):
            if down and y <= -0.26:
                down = False
            if down:
                y -= 0.00005
            else:
                x += 0.00005
            knife.set_target(np.array([x, y, z]), np.array([1, 0, 0, 0]))
            sim.substep()
            sim.toward_target(substeps=1)
        sim.update_scene()
        sim.show()
        # export_mesh = trimesh.Trimesh(
        #     sim.x_soft.to_numpy(), np.asarray(dumpling_mesh.faces))
        # export_mesh.export(f'./out/cut/{frame}.obj')
        frame += 1

if __name__ == "__main__":
    test_sim()