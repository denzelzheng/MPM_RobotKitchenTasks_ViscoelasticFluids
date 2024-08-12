import taichi as ti
import numpy as np
import os
from data_utils import preprocess_point_clouds, preprocess_mechanical_data
from visualize import visualize_3d_point_clouds
from simulation import ParticleSystem
from adaptive_particle_model import AdaptiveParticleModel, SurfaceToParticleModel
import open3d as o3d

VISUALIZE_MODE = 1

E = 1e3
nu = 0.4
yield_stress = 100
visco = 1e-0
end_step = 2048

num_iterations = 10
learning_rate = 1e-3

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

container_length = 0.3 
container_width = 0.2
container_height = 0.1
material_height = 0.03


class MainStateMachine:
    def __init__(self, align_method='kdtree'):
        self.align_method = align_method
        self.particle_model = None
        self.particle_system = None
        self.first_inference = True
        
        # Initialize file paths
        self.initial_surface_path = os.path.join(current_directory, "initial_surface.ply")
        self.final_surface_path = os.path.join(current_directory, "final_surface.ply")
        self.initial_cloud_path = os.path.join(current_directory, "initial_cloud.ply")
        self.final_cloud_path = os.path.join(current_directory, "final_cloud.ply")
        self.tool_cloud_path = os.path.join(current_directory, "tool_cloud.ply")
        self.tool_traj_path = os.path.join(current_directory, "tool_traj.npy")
        self.tool_force_path = os.path.join(current_directory, "tool_force.npy")


        np.save(self.tool_traj_path, np.array([[0.5, 0.7, 0.4], [0.5, 0.7, 0.6]]))
        np.save(self.tool_force_path, np.array([0.05, 0.1, 0.05]))


    def process_surface_data(self):
        print("Processing surface data...")

        initial_surface = np.array(o3d.io.read_point_cloud(self.initial_surface_path).points)
        indices = np.random.choice(initial_surface.shape[0], size=2000, replace=False)
        initial_surface = initial_surface[indices]

        final_surface = np.array(o3d.io.read_point_cloud(self.final_surface_path).points)
        indices = np.random.choice(final_surface.shape[0], size=2000, replace=False)
        final_surface = final_surface[indices]

        # self.particle_model = AdaptiveParticleModel(np.array([0.0, 0.0, 0.0]), n_particles=5000, length=container_width, width=material_height, height=container_length)
        # initial_particles = self.particle_model.update_model(initial_surface, True)
        # final_particles = self.particle_model.update_model(final_surface, True)

        self.particle_model = SurfaceToParticleModel(bottom_thickness=material_height, particle_count=5000)
        initial_particles = self.particle_model.convert(initial_surface)
        final_particles = self.particle_model.convert(final_surface)




        # Save updated particle models
        initial_particles_pcd = o3d.geometry.PointCloud()
        initial_particles_pcd.points = o3d.utility.Vector3dVector(initial_particles)
        o3d.io.write_point_cloud(self.initial_cloud_path, initial_particles_pcd)
        final_particles_pcd = o3d.geometry.PointCloud()
        final_particles_pcd.points = o3d.utility.Vector3dVector(final_particles)
        o3d.io.write_point_cloud(self.final_cloud_path, final_particles_pcd)

        
        print("Surface data processed and saved.")

    def infer(self):
        print("Starting inference...")

        # Process mechanical data
        tool_traj_sequence, tool_force_sequence = preprocess_mechanical_data(self.tool_traj_path, self.tool_force_path, end_step)
       
        self.process_surface_data()

        # Process particle model point clouds
        initial_particles, final_particles = preprocess_point_clouds(self.initial_cloud_path, self.final_cloud_path, self.align_method)
        initial_particles = initial_particles + np.array([0.5, 0.5, 0.5], np.float32)
        final_particles = final_particles + np.array([0.5, 0.5, 0.5], np.float32)
        tool_particles = preprocess_point_clouds(self.tool_cloud_path, None, self.align_method)[0] + tool_traj_sequence[0]

        if initial_particles is None or final_particles is None:
            print("Error in preprocessing particle model point clouds. Exiting.")
            return


        if self.first_inference:
            n_particles = initial_particles.shape[0]
            n_tool_particles = tool_particles.shape[0] if tool_particles is not None else 0
            print(f"Initializing particle system with {n_particles} particles and {n_tool_particles} tool particles")
            self.particle_system = ParticleSystem(n_particles, n_tool_particles, end_step, container_length, container_width, container_height)
            self.first_inference = False

        print(f"Setting particle system with object particle models and mechanical data")
        self.particle_system.initialize_objects(initial_particles, final_particles, tool_particles)
        self.particle_system.set_mechanical_motion(tool_traj_sequence, tool_force_sequence)
        self.particle_system.set_constitutive_parameters(E, nu, yield_stress, visco)
        print("Particle system initialized")

        # Run optimization
        best_viscosity, final_loss = self.particle_system.optimize_viscosity(num_iterations, end_step, optimizer_type='eebo', learning_rate=learning_rate, momentum=0.9)
        particles_np, tool_particles_np = self.particle_system.export_deformation()

        if VISUALIZE_MODE:
            input("Press any key to continue...")
            visualize_3d_point_clouds(particles_np[:end_step], tool_particles_np[:end_step], final_particles, False)

        print("Inference completed")

def main(align_method='kdtree'):
    state_machine = MainStateMachine(align_method)
    
    while True:
        state_machine.infer()

        # command = input("Enter command (infer/quit): ").lower()
        # if command == "quit":
        #     break
        # elif command == "infer":
        #     state_machine.infer()
        # else:
        #     print(f"Unknown command: {command}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        align_method = 'kdtree'
    else:
        align_method = sys.argv[1].lower()
    main(align_method)