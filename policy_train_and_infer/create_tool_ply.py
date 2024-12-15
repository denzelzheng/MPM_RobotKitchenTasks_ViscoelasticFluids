import open3d as o3d
import numpy as np
import os
import time

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# 导入OBJ模型
mesh_path = os.path.join(current_directory, "tool.obj")
print(f"Loading mesh from: {mesh_path}")
mesh = o3d.io.read_triangle_mesh(mesh_path)

print(f"Mesh loaded. Vertex count: {len(mesh.vertices)}, Triangle count: {len(mesh.triangles)}")

# 尝试修复网格
print("Repairing mesh...")
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()

# 确保法线信息存在并且方向一致
print("Computing normals...")
mesh.compute_vertex_normals()
mesh.orient_triangles()

# 使用采样方法生成点云
print("Generating point cloud...")
start_time = time.time()
num_points = 3000  # 可以调整这个值来改变点的数量
pcd = mesh.sample_points_uniformly(number_of_points=num_points)
end_time = time.time()
print(f"Point cloud generation took {end_time - start_time:.2f} seconds")

print(f"Generated {len(pcd.points)} points")

if len(pcd.points) == 0:
    print("Warning: No points generated. This might indicate an issue with the mesh or the sampling process.")
else:
    # 获取点云的边界框
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    print(f"Original bounding box: Min {min_bound}, Max {max_bound}")

    # 计算每个轴的长度
    axis_lengths = max_bound - min_bound
    print(f"Axis lengths: {axis_lengths}")

    # 找出最长的轴
    longest_axis = np.argmax(axis_lengths)
    print(f"Longest axis: {longest_axis}")

    # 计算旋转矩阵，使最长轴对齐Y轴
    rotation_matrix = np.eye(3)
    if longest_axis != 1:  # 如果最长轴不是Y轴
        # 交换最长轴和Y轴
        rotation_matrix[:, [longest_axis, 1]] = rotation_matrix[:, [1, longest_axis]]

    # 应用旋转
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    # 重新计算边界框
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    axis_lengths = max_bound - min_bound
    print(f"Bounding box after rotation: Min {min_bound}, Max {max_bound}")
    print(f"Axis lengths after rotation: {axis_lengths}")

    # 缩放点云
    target_length = 0.35  # 设置目标长度，您可以根据需要调整这个值
    scale_factor = target_length / axis_lengths[1]  # Y轴现在是最长轴
    pcd.scale(scale_factor, center=(0, 0, 0))

    # 重新计算边界框
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    center = (min_bound + max_bound) / 2

    # 平移点云使其中心在原点
    pcd.translate(-center)
    # pcd.translate(np.array([0.5, 0.5, 0.5]))

    # 最后一次重新计算边界框
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    axis_lengths = max_bound - min_bound
    print(f"Final bounding box: Min {min_bound}, Max {max_bound}")
    print(f"Final axis lengths: {axis_lengths}")

    # 可视化
    print("Visualizing...")
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云模型
    pcd.paint_uniform_color([0, 1, 1])  
    vis.add_geometry(pcd)

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0.5])
    vis.add_geometry(coordinate_frame)

    # 设置视图
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # 灰色背景
    opt.point_size = 5

    # 运行可视化
    vis.run()
    vis.destroy_window()

    # 保存粒子模型为PLY文件
    pcd_path = os.path.join(current_directory, "tool_cloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Particle model saved as 'tool_cloud.ply' with {len(pcd.points)} points")