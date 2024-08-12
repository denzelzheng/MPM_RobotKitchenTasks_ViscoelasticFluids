# data_utils.py

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import pickle
import os
from scipy import interpolate
import time

def read_point_cloud(file_path):
    print(f"Reading point cloud from {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def align_point_clouds_hungarian(source, target):
    print("Aligning point clouds using Hungarian algorithm")
    start_time = time.time()
    
    print("Calculating distance matrix")
    dist_matrix = cdist(source, target)
    
    print("Running Hungarian algorithm")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    print("Reordering target point cloud")
    aligned_target = target[col_ind]
    
    end_time = time.time()
    print(f"Hungarian alignment completed in {end_time - start_time:.2f} seconds")
    
    return aligned_target

def align_point_clouds_kdtree(source, target):
    print("Aligning point clouds using KD-tree")
    start_time = time.time()
    
    print("Building KD-tree")
    tree = cKDTree(target)
    
    print("Finding nearest neighbors")
    distances, indices = tree.query(source, k=1)
    
    print("Reordering target point cloud")
    aligned_target = target[indices]
    
    end_time = time.time()
    print(f"KD-tree alignment completed in {end_time - start_time:.2f} seconds")
    
    return aligned_target


def adjust_particle_count(points_a, points_b, target_count):
    min_count = min(len(points_a), len(points_b), target_count)
    print(f"Adjusting particle count to {min_count}")
    
    adjusted_points_a = _adjust_single_cloud(points_a, min_count)
    adjusted_points_b = _adjust_single_cloud(points_b, min_count)
    
    return adjusted_points_a, adjusted_points_b

def _adjust_single_cloud(points, target_count):
    if len(points) == target_count:
        return points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if len(points) < target_count:
        pcd_dense = pcd.uniform_down_sample(1)
        pcd_dense = pcd_dense.voxel_down_sample(voxel_size=0.01)
        while len(pcd_dense.points) < target_count:
            pcd_dense = pcd_dense.uniform_down_sample(1)
        adjusted_points = np.asarray(pcd_dense.points)[:target_count]
    else:
        pcd_down = pcd.uniform_down_sample(len(points) // target_count)
        adjusted_points = np.asarray(pcd_down.points)[:target_count]
    
    return adjusted_points



def calculate_total_distance(source, target):
    """
    Calculate the total Euclidean distance between two point clouds.
    
    Args:
    source (np.array): The source point cloud, shape (n, 3)
    target (np.array): The target point cloud, shape (n, 3)
    
    Returns:
    float: The total Euclidean distance between the two point clouds
    """
    # Ensure the point clouds have the same number of points
    assert source.shape == target.shape, "Point clouds must have the same shape"
    
    # Calculate the Euclidean distance for each pair of points
    distances = np.sqrt(np.sum((source - target)**2, axis=1))
    
    # Sum up all the distances
    total_distance = np.sum(distances)
    
    return total_distance

def preprocess_point_clouds(initial_cloud_path, final_cloud_path, align_method='kdtree'):
    print("Starting point cloud preprocessing")
    
    initial_points = read_point_cloud(initial_cloud_path)
    print(f"Initial points shape: {initial_points.shape}, dtype: {initial_points.dtype}")
    
    if initial_points.dtype != np.float32:
        initial_points = initial_points.astype(np.float32)
        print("Converted initial points to float32")
    
    if final_cloud_path is not None:
        final_points = read_point_cloud(final_cloud_path)
        print(f"Final points shape: {final_points.shape}, dtype: {final_points.dtype}")
        
        if final_points.dtype != np.float32:
            final_points = final_points.astype(np.float32)
            print("Converted final points to float32")
        
        if align_method != 'none':
            print(f"Aligning point clouds using {align_method} method")
            if align_method == 'hungarian':
                aligned_final_points = align_point_clouds_hungarian(initial_points, final_points)
            elif align_method == 'kdtree':
                aligned_final_points = align_point_clouds_kdtree(initial_points, final_points)
            else:
                print(f"Unknown alignment method: {align_method}. Using KD-tree method.")
                aligned_final_points = align_point_clouds_kdtree(initial_points, final_points)
            
            print(f"Aligned points shape: {aligned_final_points.shape}, dtype: {aligned_final_points.dtype}")
            
            # Ensure aligned points are float32
            if aligned_final_points.dtype != np.float32:
                aligned_final_points = aligned_final_points.astype(np.float32)
                print("Converted aligned points to float32")
            
            # Calculate and print the total distance after alignment
            total_distance = calculate_total_distance(initial_points, aligned_final_points)
            print(f"Total distance after alignment: {total_distance:.4f}")
            
            print("Saving aligned points to temporary file")
            with open('tmp_aligned_points.pkl', 'wb') as f:
                pickle.dump(aligned_final_points, f)
        else:
            print("Skipping alignment, trying to load from temporary file")
            try:
                with open('tmp_aligned_points.pkl', 'rb') as f:
                    aligned_final_points = pickle.load(f)
                print("Loaded aligned points from temporary file")
                print(f"Loaded points shape: {aligned_final_points.shape}, dtype: {aligned_final_points.dtype}")
                
                # Ensure loaded points are float32
                if aligned_final_points.dtype != np.float32:
                    aligned_final_points = aligned_final_points.astype(np.float32)
                    print("Converted loaded points to float32")
                
                # Calculate and print the total distance for loaded points
                total_distance = calculate_total_distance(initial_points, aligned_final_points)
                print(f"Total distance for loaded points: {total_distance:.4f}")
            except FileNotFoundError:
                print("Temporary file not found. Please run with alignment first.")
                return None, None
        
        print("Point cloud preprocessing completed")
        return initial_points, aligned_final_points
    else:
        print("No final cloud provided. Returning only initial points.")
        return initial_points, None

def preprocess_mechanical_data(traj_path, force_path, max_steps):
    # Load data
    tool_traj_sequence = np.load(traj_path)
    tool_force_sequence = np.load(force_path)
    
    print(f"Loaded trajectory data shape: {tool_traj_sequence.shape}, dtype: {tool_traj_sequence.dtype}")
    print(f"Loaded force data shape: {tool_force_sequence.shape}, dtype: {tool_force_sequence.dtype}")
    
    # Convert to float32 if needed
    if tool_traj_sequence.dtype != np.float32:
        tool_traj_sequence = tool_traj_sequence.astype(np.float32)
        print("Converted trajectory data to float32")
    if tool_force_sequence.dtype != np.float32:
        tool_force_sequence = tool_force_sequence.astype(np.float32)
        print("Converted force data to float32")
    
    # Ensure trajectory and force data have the same length
    assert len(tool_traj_sequence) == len(tool_force_sequence), "Mismatch in trajectory and force data lengths"
    
    original_steps = len(tool_traj_sequence)
    print(f"Original number of steps: {original_steps}")
    
    if original_steps == max_steps:
        print("Data already matches required steps. No preprocessing needed.")
        return tool_traj_sequence, tool_force_sequence
    
    # Create index arrays for original and target steps
    original_indices = np.arange(original_steps, dtype=np.float32)
    target_indices = np.linspace(0, original_steps - 1, max_steps, dtype=np.float32)
    
    # Interpolate or sample trajectory data
    traj_interpolator = interpolate.interp1d(original_indices, tool_traj_sequence, axis=0, kind='linear')
    new_tool_traj = traj_interpolator(target_indices).astype(np.float32)
    
    # Interpolate or sample force data
    force_interpolator = interpolate.interp1d(original_indices, tool_force_sequence, kind='linear')
    new_tool_force = force_interpolator(target_indices).astype(np.float32)
    
    print(f"Preprocessed trajectory data shape: {new_tool_traj.shape}, dtype: {new_tool_traj.dtype}")
    print(f"Preprocessed force data shape: {new_tool_force.shape}, dtype: {new_tool_force.dtype}")
    
    if original_steps < max_steps:
        print(f"Interpolated data from {original_steps} to {max_steps} steps")
    else:
        print(f"Sampled data from {original_steps} to {max_steps} steps")
    
    return new_tool_traj, new_tool_force