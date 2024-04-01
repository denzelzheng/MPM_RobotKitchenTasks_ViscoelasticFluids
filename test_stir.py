import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim import StaticBoundary, DynamicBoundary, RigidBody, SoftBody, MpmSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh, rotate_mesh
from sim import NeoHookean, StVK_with_Hecky_strain, visco_StVK_with_Hecky_strain

def test_sim():
    ti.init(arch=ti.cuda)
    dt = 1e-4

    substeps = 5


    stir_folder = './data/stir/'
    cut_folder = './data/cut/cut0001/'
    equilibrated_material = StVK_with_Hecky_strain(3, 0.15, True)
    non_equilibrated_material = visco_StVK_with_Hecky_strain(3, 0.15, 1e-4, 1e-4, False)

    fluid_par = np.random.rand(30000, 3) * 0.15
    fluid_par = fluid_par - fluid_par.mean(axis=0) + np.array([0.0, -0.01, 0.0])
    non_equilibrated_fluid = SoftBody(fluid_par, non_equilibrated_material)
    equilibrated_fluid = SoftBody(fluid_par, equilibrated_material)

    chopping_board_mesh = trimesh.load_mesh(pjoin(cut_folder, 'chopping_board.obj'))
    chopping_board_mesh.vertices += np.array([0.5, 0.4, 0.5])
    chopping_board = StaticBoundary(mesh=chopping_board_mesh)

    basin_mesh = trimesh.load_mesh(pjoin(stir_folder, 'basin_remesh1.obj'))
    basin_mesh.vertices += -basin_mesh.vertices.mean(axis=0)
    basin_mesh.vertices += np.array([0.5, 0.45, 0.5])
    basin = DynamicBoundary(mesh=basin_mesh)

    basin_mesh_lag = trimesh.load_mesh(pjoin(stir_folder, 'basin_remesh1.obj'))
    basin_mesh_lag.vertices = basin_mesh.vertices - basin_mesh.vertices.mean(axis=0)
    # basin_mesh_lag.vertices *= 0.5
    basin_mesh_lag.vertices += np.array([0.0, -0.06, 0.0])

    
    shovel_mesh = trimesh.load_mesh(pjoin(stir_folder, 'shovel_remesh3.obj'))
    # for m in shovel_mesh.geometry.values():
    #     m.vertices += np.array([0.5, 0.7, 0.5])
    shovel_mesh.vertices += -shovel_mesh.vertices.mean(axis=0)
    shovel_mesh.vertices *= 1.5
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'x', -90)
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'y', -45)
    shovel_mesh.vertices += np.array([0.5, 0.7, 0.5])
    shovel = DynamicBoundary(mesh=shovel_mesh, collide_type="both")
    # shovel = StaticBoundary(mesh=shovel_mesh)

    sim = MpmSim(origin=np.asarray([-0.5,] * 3), dt=dt, ground_friction=0, box_bound_rel=0.1)
    sim.set_camera_pos(0.5, 0.8, 1)
    # sim.set_camera_pos(0.5, 1.3, 1.5)
    sim.camera_lookat(0.5, 0.5, 0.5)
    sim.add_boundary(chopping_board)
    sim.add_boundary(shovel)
    # sim.add_boundary(basin)
    sim.add_lag_body(basin_mesh_lag, 1.5e4, 0.1)
    sim.add_body(non_equilibrated_fluid)
    sim.add_body(equilibrated_fluid)
    sim.init_system()

    print("start simulation...")
    print("({} static and {} dynamic boundary)".format(sim.n_static_bounds, sim.n_dynamic_bounds))
    frame = 0
    x, y, z = 0, 0, 0
    down = True
    stir_right = True
    stir_limit = 0.07
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        for s in range(substeps):
            # 下降逻辑
            if down and y <= -0.15:
                down = False  
            if down:
                y -= 0.00005
            else:  # 搅拌逻辑
                if stir_right and x >= stir_limit: 
                    stir_right = False 
                elif not stir_right and x <= -stir_limit:  
                    stir_right = True 
                if stir_right:
                    x += 0.00005 
                else:
                    x -= 0.00005 
            shovel.set_target(np.array([x, y, z]), np.array([1, 0, 0, 0]))
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