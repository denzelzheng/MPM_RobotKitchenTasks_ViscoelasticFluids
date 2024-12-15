import numpy as np
import open3d as o3d
import time

lower_bound = 0
upper_bound = 1 - 0
points = [
    [lower_bound, lower_bound, lower_bound],
    [upper_bound, lower_bound, lower_bound],
    [lower_bound, upper_bound, lower_bound],
    [upper_bound, upper_bound, lower_bound],
    [lower_bound, lower_bound, upper_bound],
    [upper_bound, lower_bound, upper_bound],
    [lower_bound, upper_bound, upper_bound],
    [upper_bound, upper_bound, upper_bound],
]
lines = [
    [0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6],
    [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7],
]
# colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)

import numpy as np
import open3d as o3d
import time

def visualize_3d_point_clouds(arr1, arr2, target_arr, first_frame_only=False, target_alpha=0.3):
    # 创建三个点云对象
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd_target = o3d.geometry.PointCloud()

    # 更新第一帧的点云
    points1 = np.array(arr1[0]).reshape(-1, 3)
    points2 = np.array(arr2[0]).reshape(-1, 3)
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2.points = o3d.utility.Vector3dVector(points2)

    # 设置目标点云
    points_target = np.array(target_arr).reshape(-1, 3)
    pcd_target.points = o3d.utility.Vector3dVector(points_target)

    # 设置点云颜色
    pcd1.paint_uniform_color([1, 0, 0])  # 红色
    pcd2.paint_uniform_color([0, 0, 1])  # 蓝色
    # 设置目标点云颜色（半透明绿色）
    target_color = np.array([0, 1, 0, target_alpha])  # 绿色带透明度
    pcd_target.colors = o3d.utility.Vector3dVector(np.tile(target_color[:3], (len(points_target), 1)))

    # 创建坐标系
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=(0., 0., 0.))

    if first_frame_only:
        # 使用 draw_geometries 显示第一帧
        o3d.visualization.draw_geometries([coordinate, pcd1, pcd2, pcd_target, line_set], lookat=[0, 0, 0], up=[0, 1, 0], front=[0, 0, -1], zoom=0.8)
    else:
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 将几何体添加到可视化器
        vis.add_geometry(coordinate)
        vis.add_geometry(line_set)
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2)
        vis.add_geometry(pcd_target)

        # 设置视图
        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_front([0, 0, -1])
        ctr.set_zoom(0.8)

        # 获取帧数
        num_frames = max(len(arr1), len(arr2))

        # 动画循环
        for frame in range(num_frames):
            # 更新第一个点云
            if frame < len(arr1):
                points1 = np.array(arr1[frame]).reshape(-1, 3)
                pcd1.points = o3d.utility.Vector3dVector(points1)

            # 更新第二个点云
            if frame < len(arr2):
                points2 = np.array(arr2[frame]).reshape(-1, 3)
                pcd2.points = o3d.utility.Vector3dVector(points2)

            # 更新几何体
            vis.update_geometry(pcd1)
            vis.update_geometry(pcd2)
            vis.poll_events()
            vis.update_renderer()

            # 控制帧率
            time.sleep(0.01)  

        vis.run()
        # 关闭可视化窗口
        vis.destroy_window()

