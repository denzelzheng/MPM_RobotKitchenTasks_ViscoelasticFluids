import numpy as np
import plotly.graph_objects as go
from adaptive_particle_model import AdaptiveParticleModel, SurfaceToParticleModel
import os

surface_points = np.random.rand(1000, 3)

# Create converter instance
converter = SurfaceToParticleModel(bottom_thickness=0.1, particle_count=2000)

# Convert surface point cloud to particle model
particle_model = converter.convert(surface_points)

# Use update_parameters method if parameters need to be updated
particle_model = converter.update_parameters(surface_points, bottom_thickness=0.2, particle_count=3000)

def generate_single_side_surface(n_points, length, width):
    """Generate irregular surface point cloud for one side"""
    x = np.random.rand(n_points) * length
    y = np.random.rand(n_points) * width
    z = np.random.normal(0, 0.1, n_points) + max(length, width)  # Keep z values concentrated near a plane
    return np.column_stack((x, y, z))

def visualize_particles(particles, surface_points, title):
    """Visualize particles and surface points"""
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

# Main program
if __name__ == "__main__":
    # Define cuboid dimensions
    length, width, height = 1.0, 0.8, 0.6

    # Initialize particle model
    model = SurfaceToParticleModel(bottom_thickness=0.1, particle_count=2000)

    # Generate initial irregular surface point cloud
    initial_surface = generate_single_side_surface(100, length, width)

    # Update model to adapt to initial surface
    updated_particles = model.convert(initial_surface)

    # Visualize results
    visualize_particles(updated_particles, initial_surface, "Initial Adaptation")

    # Simulate receiving new surface point clouds and updating the model
    for i in range(2):  # Simulate 3 updates
        new_surface = generate_single_side_surface(1000, length, width)
        updated_particles = model.convert(new_surface)

        visualize_particles(updated_particles, new_surface, f"Update {i+1}")