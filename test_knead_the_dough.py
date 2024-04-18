import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim_w_mix_and_hydration import StaticBoundary, DynamicBoundary, RigidBody, SoftBody, MpmSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh, rotate_mesh
from sim_w_mix_and_hydration import NeoHookean, StVK_with_Hecky_strain, visco_StVK_with_Hecky_strain, \
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
    dough_material = hydration_material(13, 0.01, 0.45, 0.7, False)
    flour_par = np.random.rand(10000, 3) * 0.15 * np.array([1, 1, 1])
    flour_par = flour_par - flour_par.mean(axis=0) + np.array([0.0, -0.01, 0.0])
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
    basin_mesh_lag.vertices += np.array([0.0, -0.06, 0.0])  # type: ignore


    cup_pos = np.array([0.73, 0.45, 0.55])
    cup_mesh = trimesh.load_mesh(pjoin(stir_folder, 'basin_remesh1.obj'))
    cup_mesh = thicken_mesh(cup_mesh, 3e-3)
    cup_mesh.vertices = cup_mesh.vertices - \
        cup_mesh.vertices.mean(axis=0)
    cup_mesh.vertices *= np.array([0.25, 1.3, 0.25])
    cup_mesh.vertices += cup_pos
    cup = DynamicBoundary(mesh=cup_mesh, collide_type="both")
    # cup2 = DynamicBoundary(mesh=cup_mesh, collide_type="both")
    

    water_material = NeoHookean(3e-2, 0.45, True)
    water_par = np.random.rand(15000, 3) * 0.05 * np.array([1, 1.5, 1])
    water_par = water_par - water_par.mean(axis=0) + cup_pos - np.array([0.5, 0.5, 0.5])
    water_color = np.array([0.45, 0.45, 1])
    water = SoftBody(
        water_par, water_material, water_color, 0.0, 1.0, 0.9)


    shovel_mesh = trimesh.load_mesh(pjoin(stir_folder, 'shovel_remesh3.obj'))
    # for m in shovel_mesh.geometry.values():
    #     m.vertices += np.array([0.5, 0.7, 0.5])
    shovel_mesh.vertices += -shovel_mesh.vertices.mean(axis=0) # type: ignore
    shovel_mesh.vertices *= 1.5
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'x', -90)
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'y', -45)
    shovel_pos = np.array([0.5, 0.75, 0.5])
    shovel_mesh.vertices += shovel_pos
    shovel = DynamicBoundary(mesh=shovel_mesh, collide_type="both")
    # shovel = StaticBoundary(mesh=shovel_mesh)

    sim = MpmSim(origin=np.asarray([-0.5, ] * 3),
                 dt=dt, ground_friction=0, box_bound_rel=0.1)
    sim.set_camera_pos(0.31, 1, 0.75)
    # sim.set_camera_pos(0.5, 1.3, 1.5)
    sim.camera_lookat(0.5, 0.5, 0.5)
    sim.add_boundary(chopping_board)
    sim.add_boundary(shovel)
    sim.add_boundary(cup)
    # sim.add_boundary(cup2)
    sim.add_lag_body(basin_mesh_lag, 5e4, 0.1)
    sim.add_body(flour)
    sim.add_body(water)
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

    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        
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


            if obj2_up and obj2_y >= 0.13:
                obj2_up = False
            if obj2_up:
                obj2_y += 0.00002
            elif obj2_pour and obj2_move_to_left_limit:
                obj2_x -= 0.00003
            if obj2_x <= obj2_left_limit:
                obj2_move_to_left_limit = False
                if pour_angle < 155 and obj2_pour:
                    pour_angle += pour_angle_step
                elif pour_angle >= 155:
                    obj2_pour = False
                if obj2_pour==False and pour_angle>0:
                    pour_angle -= pour_angle_step
            if obj2_pour==False and obj2_x < 0 and pour_angle < 1:
                obj2_x += 0.00007                    

            radian_angle = np.deg2rad(pour_angle)
            cup.set_target(np.array([obj2_x, obj2_y, obj2_z]), np.array([np.cos(radian_angle/2), 0, 0, np.sin(radian_angle/2)]), cup_pos)
            # cup2.set_target(np.array([obj2_x, obj2_y, obj2_z]), np.array([np.cos(radian_angle/2), 0, 0, np.sin(radian_angle/2)]), cup_pos)





            sim.substep()
            sim.toward_target(substeps=1)
        sim.update_scene()
        sim.show()
        frame += 1


if __name__ == "__main__":
    test_sim()
