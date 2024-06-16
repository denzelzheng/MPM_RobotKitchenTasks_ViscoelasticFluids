import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim_w_mix_and_hydration_valve import StaticBoundary, DynamicBoundary, RigidBody, SoftBody, MpmSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh, rotate_mesh
from sim_w_mix_and_hydration_valve import NeoHookean, StVK_with_Hecky_strain, visco_StVK_with_Hecky_strain, \
    cross_visco_StVK_with_Hecky_strain, NeoHookean_VonMise, hydration_material



def thicken_mesh(mesh, thickness):
    vertex_normals = mesh.vertex_normals
    outer_vertices = mesh.vertices + vertex_normals * thickness
    outer_mesh = trimesh.Trimesh(vertices=outer_vertices, faces=mesh.faces)
    combined_mesh = trimesh.util.concatenate(mesh, outer_mesh)
    return combined_mesh


def test_sim():
    ti.init(arch=ti.cuda)
    dt = 1e-4

    substeps = 5

    stir_folder = './data/stir/'
    cut_folder = './data/cut/cut0001/'
    # flour_material = NeoHookean_VonMise(15, 0.01, 0.45, False)
    # dough_material = visco_StVK_with_Hecky_strain(15, 0.01, 0.7, False)
    dough_material = hydration_material(15, 0.01, 0.45, 0.7, False)
    flour_par = np.random.rand(10000, 3) * 0.15 * np.array([1, 1, 1])
    flour_par = flour_par - flour_par.mean(axis=0) + np.array([0.5, 0.45, 0.5])
    flour_color = np.array([0.6, 0.6, 0.7])
    flour = SoftBody(
        flour_par, dough_material, flour_color, 1.0, 0.0, 0.9)
    
    
   

    chopping_board_mesh = trimesh.load_mesh(
        pjoin(cut_folder, 'chopping_board.obj'))
    chopping_board_mesh.vertices += np.array([0.5, 0.4, 0.5])
    chopping_board = StaticBoundary(mesh=chopping_board_mesh)

    # basin_mesh = trimesh.load_mesh(pjoin(stir_folder, 'basin_remesh1.obj'))
    # basin_mesh.vertices += -basin_mesh.vertices.mean(axis=0)  # type: ignore
    # basin_mesh.vertices += np.array([0.5, 0.45, 0.5])  # type: ignore

    basin_mesh_lag = trimesh.load_mesh(pjoin(stir_folder, 'basin_remesh1.obj'))
    basin_mesh_lag.vertices = basin_mesh_lag.vertices - \
        basin_mesh_lag.vertices.mean(axis=0)  # type: ignore
    basin_mesh_lag.vertices *= np.array([1, 1.7, 1])
    # basin_mesh_lag.vertices *= 0.5
    basin_mesh_lag.vertices += np.array([0.5, 0.44, 0.5])  # type: ignore


    water_material = NeoHookean(3e-2, 0.45, True)
    water_par = np.random.rand(15000, 3) * 0.045 * np.array([1, 1, 1])
    water_par = water_par - water_par.mean(axis=0) + np.array([0.43, 0.6, 0.5])
    water_color = np.array([0.45, 0.45, 0.7])
    water = SoftBody(
        water_par, water_material, water_color, 0.0, 1.0, 0.9)


    shovel_mesh = trimesh.load_mesh(pjoin(stir_folder, 'shovel_remesh3.obj'))
    shovel_mesh.vertices += -shovel_mesh.vertices.mean(axis=0) # type: ignore
    shovel_mesh.vertices *= 1.5
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'x', -90)
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'y', -45)
    shovel_pos = np.array([0.5, 0.75, 0.5])
    shovel_mesh.vertices += shovel_pos
    shovel = DynamicBoundary(mesh=shovel_mesh, collide_type="both")
    # shovel = StaticBoundary(mesh=shovel_mesh)

    sim = MpmSim(origin=np.asarray([0, ] * 3),
                 dt=dt, ground_friction=0, box_bound_rel=0.1)
    sim.set_camera_pos(0.31, 1, 0.75)
    # sim.set_camera_pos(0.5, 1.3, 1.5)
    sim.camera_lookat(0.5, 0.5, 0.5)
    sim.add_boundary(chopping_board)
    sim.add_boundary(shovel)
    # sim.add_boundary(cup)
    # sim.add_boundary(cup2)
    sim.add_lag_body(basin_mesh_lag, 4e4, 0.1)
    sim.add_body(flour)
    sim.add_body(water)
    # sim.add_body(cup_mac)
    sim.init_system()

    print("start simulation...")
    print("({} static and {} dynamic boundary)".format(
        sim.n_static_bounds, sim.n_dynamic_bounds))

    frame = 0
    obj1_x, obj1_y, obj1_z = 0, 0, 0
    obj1_down = True
    obj1_move_to_right_limit = True
    stir_limit = 0.066
    circle_radius = stir_limit
    angle = 0
    # angle_step = 0.0015
    angle_step = 0.0021
    circle_center_x = 0
    circle_center_z = 0
    obj1_y_v = 0.00008
    obj1_x_v = 0.00005

    obj2_x, obj2_y, obj2_z = 0, 0, 0
    obj2_up = True
    obj2_move_to_left_limit = True
    obj2_left_limit = -0.153
    obj2_pour = True

    pour_angle = 0
    pour_angle_step = 0.015


    valve = 0.575
    
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        
        sim.set_valve(valve)
        # print("valve", valve)
        for i in range(1000):
            for s in range(substeps):
                if obj2_x > -0.1:
                    if obj1_down and obj1_y <= -0.215:
                        obj1_down = False
                    if obj1_down:
                        obj1_y -= obj1_y_v
                    elif obj1_move_to_right_limit:
                        if obj1_x >= stir_limit:
                            obj1_move_to_right_limit = False
                            circle_center_x = 0
                            circle_center_z = 0
                        else:
                            obj1_x += obj1_x_v
                    else:
                        obj1_x = circle_radius * np.cos(angle) + circle_center_x
                        obj1_z = circle_radius * np.sin(angle) + circle_center_z
                        angle += angle_step
                shovel.set_target(np.array([obj1_x, obj1_y, obj1_z]), np.array([1, 0, 0, 0]), shovel_pos)
                sim.substep()
                sim.toward_target(substeps=1)
            sim.update_scene()
            sim.show()
            frame += 1
        if valve <= 0.62:
            valve += 0.005


if __name__ == "__main__":
    test_sim()
