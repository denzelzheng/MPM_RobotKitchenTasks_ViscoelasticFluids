import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim_w_mix_and_hydration_RL import StaticBoundary, DynamicBoundary, RigidBody, SoftBody, MpmSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh, rotate_mesh
from sim_w_mix_and_hydration_RL import NeoHookean, StVK_with_Hecky_strain, visco_StVK_with_Hecky_strain, \
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
    n_grids = 128

    substeps = 5

    scale = 1.05

    stir_folder = './data/stir/'
    cut_folder = './data/cut/cut0001/'
    # flour_material = NeoHookean_VonMise(15, 0.01, 0.45, False)
    # dough_material = visco_StVK_with_Hecky_strain(15, 0.01, 0.7, False)

    dough_material = hydration_material(1500, 0.01, 0.15, 6800, 2.5, False)
    flour_par = np.random.rand(15000, 3) * 0.13 * np.array([1.1, 0.37, 0.5]) * scale
    flour_par = flour_par - flour_par.mean(axis=0) + np.array([0.5, 0.47, 0.5])
    flour_color = np.array([0.6, 0.6, 0.7])
    flour = SoftBody(
        flour_par, dough_material, flour_color, 1.0, 0.0, 0.9)
    

    chopping_board_mesh = trimesh.load_mesh(
        pjoin(cut_folder, 'chopping_board.obj'))
    chopping_board_mesh.vertices *= scale
    # chopping_board_mesh.vertices += -chopping_board_mesh.vertices.mean(axis=0)
    chopping_board_mesh.vertices += np.array([0.5, 0.40, 0.5])
    # chopping_board = StaticBoundary(mesh=chopping_board_mesh)
    chopping_board = DynamicBoundary(mesh=chopping_board_mesh, collide_type="grid")

    # basin_mesh = trimesh.load_mesh(pjoin(stir_folder, 'basin_remesh1.obj'))
    # basin_mesh.vertices += -basin_mesh.vertices.mean(axis=0)  # type: ignore
    # basin_mesh.vertices += np.array([0.5, 0.45, 0.5])  # type: ignore

    basin_mesh_lag = trimesh.load_mesh(pjoin(stir_folder, 'box_remesh1.obj'))
    basin_mesh_lag.vertices = basin_mesh_lag.vertices - \
        basin_mesh_lag.vertices.mean(axis=0)
    basin_mesh_lag.vertices *= np.array([0.95, 1.2, 0.5]) * scale
    basin_mesh_lag.vertices += np.array([0.5, 0.45, 0.5]) 

    water_material = NeoHookean(5e-6, 0.45, True)
    water_par = np.random.rand(10000, 3) * 0.045 * np.array([1.6, 0.25, 1.3])
    water_par = water_par - water_par.mean(axis=0)  * scale
    water_par += np.array([0.5, 0.58, 0.5])
    water_color = np.array([0.45, 0.45, 0.7])
    water = SoftBody(
        water_par, water_material, water_color, 0.0, 1.0, 0.9)


    shovel_mesh = trimesh.load_mesh(pjoin(stir_folder, 'shovel_remesh3.obj'))
    shovel_mesh.vertices += -shovel_mesh.vertices.mean(axis=0) # type: ignore
    shovel_mesh.vertices *= scale
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'x', -90)
    shovel_mesh.vertices = rotate_mesh(shovel_mesh.vertices, 'y', -90)
    shovel_mesh.vertices *= np.array([1.5, 1.5, 1.8])
    shovel_pos = np.array([0.5, 0.75, 0.5])
    shovel_mesh.vertices += shovel_pos
    shovel = DynamicBoundary(mesh=shovel_mesh, collide_type="both")

    sim = MpmSim(origin=np.asarray([0, ] * 3),
                 dt=dt, ground_friction=0, box_bound_rel=0.1, n_grids=n_grids)
    sim.set_camera_pos(0.31, 1, 0.75)
    # sim.set_camera_pos(1, 0.55, 1.5) # side view
    sim.camera_lookat(0.5, 0.5, 0.5)
    sim.add_boundary(chopping_board)
    sim.add_boundary(shovel)

    sim.add_lag_body(basin_mesh_lag, 1e8, 0.1)



    sim.add_body(flour)
    sim.add_body(water)
    sim.init_system()

    print("start simulation...")
    print("({} static and {} dynamic boundary)".format(
        sim.n_static_bounds, sim.n_dynamic_bounds))

    frame = 0
    obj1_x, obj1_y, obj1_z = 0, 0, 0
    obj1_down = True
    obj1_right = True

    obj1_up = False
    obj1_left = False
    obj1_in_work_zone = False

    push_pos = 0.055
    dig_pos = 0.098
 
    lift_height = -0.065
    stir_depth = -0.185

    obj1_y_v = 0.0003
    obj1_x_v = 0.00015

    obj2_x, obj2_y, obj2_z = 0, 0, 0

    valve = 0.573
    
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        
        sim.set_valve(valve)
        # print("valve", valve)
        for i in range(1000):
            for s in range(substeps):

                
                if not obj1_in_work_zone:
                    obj1_down = True
                    obj1_left = False
                    obj1_right = False
                    if obj1_y <= lift_height:
                        obj1_in_work_zone = True
                        obj1_left = True

                if obj1_y < stir_depth:
                    if obj1_x < -push_pos:
                        if not obj1_right:
                            obj1_up = True
                    if obj1_x > push_pos: 
                        if not obj1_left:
                            obj1_up = True
                
                if obj1_y > lift_height and obj1_in_work_zone:
                    if obj1_x < -dig_pos:
                        obj1_down = True
                        obj1_right = True
                        obj1_left = False
                    if obj1_x > dig_pos:
                        obj1_down = True
                        obj1_left = True
                        obj1_right = False


                if obj1_down:
                    if obj1_y >= stir_depth:
                        obj1_y -= obj1_y_v
                    else:
                        obj1_down = False

                if obj1_up:
                    if obj1_y <= lift_height:
                        obj1_y += obj1_y_v
                    else:
                        obj1_up = False
                
                if obj1_y <= stir_depth:
                    if obj1_left:
                        if obj1_x >= -push_pos:
                            obj1_x -= obj1_x_v
                    if obj1_right:
                        if obj1_x <= push_pos:
                            obj1_x += obj1_x_v

                if obj1_y >= lift_height:
                    if obj1_left:
                        if obj1_x >= -dig_pos:
                            obj1_x -= obj1_x_v
                        else:
                            obj1_left = False
                            obj1_right = True
                    if obj1_right:
                        if obj1_x <= dig_pos:
                            obj1_x += obj1_x_v
                        else:
                            obj1_right = False
                            obj1_left = True


    
                shovel.set_target(np.array([obj1_x, obj1_y, obj1_z]), np.array([1, 0, 0, 0]), shovel_pos)
                sim.substep()
                sim.toward_target(substeps=1)
            sim.update_scene()
            sim.show()
            frame += 1
        if valve <= 0.62:
            valve += 0.0015


if __name__ == "__main__":
    test_sim()
