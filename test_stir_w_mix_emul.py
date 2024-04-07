import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim_w_mix_and_emul import StaticBoundary, DynamicBoundary, RigidBody, SoftBody, MpmSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh, rotate_mesh
from sim_w_mix_and_emul import NeoHookean, StVK_with_Hecky_strain, visco_StVK_with_Hecky_strain, \
    cross_visco_StVK_with_Hecky_strain


def test_sim():
    ti.init(arch=ti.cuda)
    dt = 1e-4

    substeps = 5

    stir_folder = './data/stir/'
    cut_folder = './data/cut/cut0001/'
    equilibrated_material = StVK_with_Hecky_strain(0.3, 0.15, True)
    non_equilibrated_material = cross_visco_StVK_with_Hecky_strain(0.3, 0.15, 0.01, 0.01, 1, 1, True)

    fluid_par = np.random.rand(15000, 3) * 0.15
    fluid_par = fluid_par - fluid_par.mean(axis=0) + np.array([0.0, -0.01, 0.0])
    fluid_par1 = np.random.rand(15000, 3) * 0.15
    fluid_par1 = fluid_par1 - fluid_par1.mean(axis=0) + np.array([0.0, -0.01, 0.0])
    non_equilibrated_fluid = SoftBody(fluid_par, non_equilibrated_material, np.array([0.85, 0.65, 0.1]), 1, 0, 0.9)
    equilibrated_fluid = SoftBody(fluid_par1, equilibrated_material, np.array([0.85, 0.65, 0.1]), 1, 0, 0.9)

    fluid_par2 = np.random.rand(5000, 3) * 0.05
    fluid_par2 = fluid_par2 - fluid_par2.mean(axis=0) + np.array([0.0, 0.1, 0.0])
    yolk_material = cross_visco_StVK_with_Hecky_strain(0.5, 0.15, 0.1, 0.1, 1, 1)
    yolk = SoftBody(rest_pars_pos=fluid_par2, material=yolk_material, color=np.array([1.0, 0.3, 0.0]),  
                     emulsification_efficiency = 0, emulsifier_capacity = 1, density=1)


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
    shovel_mesh.vertices += np.array([0.5, 0.75, 0.5])
    shovel = DynamicBoundary(mesh=shovel_mesh, collide_type="both")
    # shovel = StaticBoundary(mesh=shovel_mesh)

    sim = MpmSim(origin=np.asarray([-0.5,] * 3), dt=dt, ground_friction=0, box_bound_rel=0.1)
    sim.set_camera_pos(0.5, 0.8, 1)
    # sim.set_camera_pos(0.5, 1.3, 1.5)
    sim.camera_lookat(0.5, 0.5, 0.5)
    sim.add_boundary(chopping_board)
    sim.add_boundary(shovel)
    # sim.add_boundary(basin)
    sim.add_lag_body(basin_mesh_lag, 5e4, 0.1)
    sim.add_body(non_equilibrated_fluid)
    sim.add_body(equilibrated_fluid)
    sim.add_body(yolk)
    sim.init_system()

    print("start simulation...")
    print("({} static and {} dynamic boundary)".format(sim.n_static_bounds, sim.n_dynamic_bounds))

    frame = 0
    x, y, z = 0, 0, 0
    down = True
    move_to_right_limit = True 
    stir_limit = 0.07
    circle_radius = stir_limit
    angle = 0
    angle_step = 0.003 

    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        for s in range(substeps):
            if down and y <= -0.213:
                down = False
            if down:
                y -= 0.00005
            elif move_to_right_limit:
                if x >= stir_limit:
                    move_to_right_limit = False
                    circle_center_x = 0
                    circle_center_z = 0
                else:
                    x += 0.00005
            else:
                x = circle_radius * np.cos(angle) + circle_center_x
                z = circle_radius * np.sin(angle) + circle_center_z
                angle += angle_step
            shovel.set_target(np.array([x, y, z]), np.array([1, 0, 0, 0]))
            sim.substep()
            sim.toward_target(substeps=1)
        sim.update_scene()
        sim.show()
        frame += 1

if __name__ == "__main__":
    test_sim()