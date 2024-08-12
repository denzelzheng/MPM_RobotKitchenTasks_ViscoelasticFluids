# main.py

import numpy as np
import plotly.graph_objects as go
from adaptive_particle_model import AdaptiveParticleModel
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

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
    model = AdaptiveParticleModel(np.array([0.0, 0.0, 0.0]), n_particles=5000, length=length, width=width, height=height)

    # 生成初始的不规则表面点云
    initial_surface = generate_single_side_surface(100, length, width)

    # 更新模型以适应初始表面
    updated_particles = model.update_model(initial_surface)

    # 可视化结果
    visualize_particles(updated_particles, initial_surface, "Initial Adaptation")

    # 模拟接收新的表面点云并更新模型
    for i in range(3):  # 模拟3次更新
        new_surface = generate_single_side_surface(100, length, width)
        updated_particles = model.update_model(new_surface, True)
        visualize_particles(updated_particles, new_surface, f"Update {i+1}")