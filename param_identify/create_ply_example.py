import numpy as np
import open3d as o3d
import os


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def create_sphere_points(radius, n_points):
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.column_stack((x, y, z))

def create_cube_points(side_length, n_points):
    points = np.random.rand(n_points, 3) * side_length - side_length/2
    return points

def create_and_save_point_cloud(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {filename}")

def main():
    n_points = 1100  
    
    cube_side_length = 0.15
    cube_points = create_cube_points(cube_side_length, n_points) + np.array([0.3, 0.3, 0.3])
    create_and_save_point_cloud(cube_points, current_directory + "/initial_cloud.ply")

    sphere_radius = 0.17
    sphere_points = create_sphere_points(sphere_radius, n_points) + np.array([0.3, 0.3, 0.3])
    create_and_save_point_cloud(sphere_points, current_directory + "/final_cloud.ply")

    cube_side_length = 0.1
    cube_points = create_cube_points(cube_side_length, 1000) + np.array([0.4, 0.4, 0.4])
    create_and_save_point_cloud(cube_points, current_directory + "/tool_cloud.ply")

if __name__ == "__main__":
    main()