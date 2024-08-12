# main.py


import numpy as np
import plotly.graph_objects as go
from adaptive_particle_model import AdaptiveParticleModel, SurfaceToParticleModel
import os

surface_points = np.random.rand(1000, 3)

# 创建转换器实例
converter = SurfaceToParticleModel(bottom_thickness=0.1, particle_count=2000)

# 转换表面点云到粒子模型
particle_model = converter.convert(surface_points)

# 如果需要更新参数，可以使用 update_parameters 方法
particle_model = converter.update_parameters(surface_points, bottom_thickness=0.2, particle_count=3000)
def generate_single_side_surface(n_points, length, width):
    """生成单侧的不规则表面点云"""
    x = np.random.rand(n_points) * length
    y = np.random.rand(n_points) * width
    z = np.random.normal(0, 0.1, n_points) + max(length, width)  # 使z值集中在一个平面附近
    return np.column_stack((x, y, z))

def visualize_particles(particles, surface_points, title):
    """可视化粒子和表面点"""
    fig = go.Figure(data=[
        go.Scatter3d(x=particles[:, 0], y=particles[:, 1], z=particles[:, 2],
                     mode='markers', marker=dict(size=3, color='red', opacity=0.8), name='Particles'),
        go.Scatter3d(x=surface_points[:, 0], y=surface_points[:, 1], z=surface_points[:, 2],
                     mode='markers', marker=dict(size=3, color='green', opacity=0.5), name='Surface Points')
    ])

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data')
    )

    fig.show()

# 主程序
if __name__ == "__main__":
    # 定义长方体的尺寸
    length, width, height = 1.0, 0.8, 0.6

    # 初始化粒子模型
    model = SurfaceToParticleModel(bottom_thickness=0.1, particle_count=2000)

    # 生成初始的不规则表面点云
    initial_surface = generate_single_side_surface(100, length, width)

    # 更新模型以适应初始表面
    updated_particles = model.convert(initial_surface)

    # 可视化结果
    visualize_particles(updated_particles, initial_surface, "Initial Adaptation")

    # 模拟接收新的表面点云并更新模型
    for i in range(2):  # 模拟3次更新
        new_surface = generate_single_side_surface(1000, length, width)
        updated_particles = model.convert(new_surface)

        visualize_particles(updated_particles, new_surface, f"Update {i+1}")