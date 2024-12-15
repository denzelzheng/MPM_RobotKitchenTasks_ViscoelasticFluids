import numpy as np
import open3d as o3d
from volume_calculator import VolumeCalculator
from infer_machine import MainStateMachine

class PhysicalFeatureExtractor:
    def __init__(self, initial_mass, additive_unit_mass, initial_volume, task_type):
        """
        Initialize the physical feature extractor
        
        Args:
            initial_mass: Initial material mass (kg)
            additive_unit_mass: Mass of one unit of additive (kg)
            initial_volume: Initial material volume (m³)
            task_type: Process type (0: hydration, 1: emulsification)
        """
        self.current_mass = initial_mass
        self.initial_mass = initial_mass
        self.additive_unit_mass = additive_unit_mass
        self.volume_calculator = VolumeCalculator(initial_volume)
        self.state_machine = MainStateMachine(align_method='kdtree')
        self.task_type = task_type
        
    def extract_features(self, policy):
        """
        Extract physical features based on policy
        
        Args:
            policy: (additive amount[0-2], mixing time[0-2], mixing speed[0-2])
            
        Returns:
            physical_features: numpy array of [mass, density, E, nu, viscosity, yield_stress]
        """
        # Update mass based on policy
        additive_amount = policy[0]  # 0:low, 1:medium, 2:high
        self.current_mass += self.additive_unit_mass * additive_amount
        
        # Process point clouds using state machine
        self.state_machine.process_surface_data()
        
        # Get point cloud data from processed files
        initial_particles = np.array(o3d.io.read_point_cloud(self.state_machine.initial_cloud_path).points)
        current_particles = np.array(o3d.io.read_point_cloud(self.state_machine.final_cloud_path).points)
        
        # Calculate volume using enhanced volume calculator
        current_volume = self.volume_calculator.calculate_volume(
            initial_particles,
            self.initial_mass,
            self.current_mass,
            self.task_type,
            current_particles
        )
        
        # Calculate density
        density = self.current_mass / current_volume
        
        # Update particle system with new mass if the method exists
        if hasattr(self.state_machine.particle_system, 'set_mass'):
            self.state_machine.particle_system.set_mass(self.current_mass)
            
        # Run inference to get constitutive parameters
        self.state_machine.infer()
        
        # Get constitutive parameters from state machine
        E = self.state_machine.particle_system.E[None]
        nu = self.state_machine.particle_system.nu[None]
        viscosity = self.state_machine.particle_system.viscosity[None]
        yield_stress = self.state_machine.particle_system.yield_stress[None]
        
        # Combine physical feature vector
        physical_features = np.array([
            self.current_mass,
            density,
            E,
            nu,
            yield_stress,
            viscosity
        ])
        
        return physical_features