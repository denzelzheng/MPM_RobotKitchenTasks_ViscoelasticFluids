import numpy as np
from sklearn.decomposition import PCA

class VolumeCalculator:
    """
    A volume calculator that uses point cloud data to estimate material volume
    based on geometric analysis and process type.
    """
    def __init__(self, initial_volume):
        self.initial_volume = initial_volume
        self.pca = PCA(n_components=3)
        
    def calculate_volume(self, initial_cloud, initial_mass, mass, task_type, current_cloud):
        """
        Calculate volume using point cloud data and material properties
        
        Args:
            initial_cloud: Initial point cloud data
            initial_mass: Initial mass of the material
            mass: Current mass of the material 
            task_type: Process type (0: hydration, 1: emulsification)
            current_cloud: Current point cloud data
            
        Returns:
            Calculated volume in cubic meters
        """
        # Fit PCA to get principal axes
        self.pca.fit(current_cloud)
        
        # Get eigenvalues representing spread along principal axes
        eigenvalues = self.pca.explained_variance_
        
        # Calculate basic bounding box volume
        bbox_volume = np.prod(np.sqrt(eigenvalues) * 2)
        
        # Calculate point density metric
        point_density = len(current_cloud) / bbox_volume
        
        if task_type == 0:  # Hydration
            # For hydration, account for micro-porosity and surface tension
            # Volume changes are minimal due to water being absorbed into existing structure
            
            # Calculate nominal expansion factor
            expansion_factor = 1.0 + 0.02 * np.log(mass / initial_mass)
            
            # Apply density-based correction
            density_correction = 1.0 - 0.01 * (point_density / 1000)
            
            # Final volume with very small variation
            volume = self.initial_volume * expansion_factor * density_correction
            
        else:  # Emulsification
            # For emulsification, volume changes proportionally with mass
            # Due to incorporation of dispersed phase
            
            # Calculate mass ratio
            mass_ratio = mass / initial_mass
            
            # Basic volume scaling
            volume_ratio = 0.95 * mass_ratio + 0.05
            
            # Apply geometric correction based on point cloud
            shape_factor = np.std(eigenvalues) / np.mean(eigenvalues)
            geometric_correction = 1.0 + 0.1 * shape_factor
            
            volume = self.initial_volume * volume_ratio * geometric_correction
            
        return volume