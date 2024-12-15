
import taichi as ti
import numpy as np
import os
from data_utils import preprocess_point_clouds, preprocess_mechanical_data
from visualize import visualize_3d_point_clouds
from simulation import ParticleSystem

E = 1e2
nu = 0.1
yield_stress = 1e6
visco = 0.1
end_step = 100


num_iterations = 30
learning_rate = 1e-19

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

container_length = 0.5 
container_width = 0.5
container_height = 0.3
material_height = 0.1


def main(align_method='kdtree'):
    print("Starting main function")
    initial_cloud_path = os.path.join(current_directory, "initial_cloud.ply")
    final_cloud_path = os.path.join(current_directory, "final_cloud.ply")
    tool_cloud_path = os.path.join(current_directory, "tool_cloud.ply")

    tool_traj_path = os.path.join(current_directory, "tool_traj.npy")
    tool_force_path = os.path.join(current_directory, "tool_force.npy")
    tool_traj_sequence, tool_force_sequence = preprocess_mechanical_data(tool_traj_path, tool_force_path, end_step)
    
    initial_points, final_points = preprocess_point_clouds(initial_cloud_path, final_cloud_path, align_method)
    tool_points = preprocess_point_clouds(tool_cloud_path, None, align_method)[0]

    if initial_points is None:
        print("Error in preprocessing initial points. Exiting.")
        return

    n_particles = initial_points.shape[0]
    n_tool_particles = tool_points.shape[0] if tool_points is not None else 0

    print(f"Initializing particle system with {n_particles} particles and {n_tool_particles} tool particles")
    particle_system = ParticleSystem(n_particles, n_tool_particles, end_step, container_length, container_width, container_height)

    
    print(f"Setting particle system with object particle models and mechanical data")
    particle_system.initialize_objects(initial_points, final_points, tool_points)
    particle_system.set_mechanical_motion(tool_traj_sequence, tool_force_sequence)
    particle_system.set_constitutive_parameters(E, nu, yield_stress, visco)
    print("Particle system initialized")

    # Run optimization
    best_viscosity, final_loss = particle_system.optimize_viscosity(num_iterations, end_step, optimizer_type='sgd', learning_rate=learning_rate, momentum=0.9)


    particles_np, tool_particles_np = particle_system.export_deformation()


    input("Press any key to continue...")
    visualize_3d_point_clouds(particles_np[:end_step], tool_particles_np[:end_step], final_points, False)

    print("Main function completed")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        align_method = 'kdtree'
    else:
        align_method = sys.argv[1].lower()
    main(align_method)