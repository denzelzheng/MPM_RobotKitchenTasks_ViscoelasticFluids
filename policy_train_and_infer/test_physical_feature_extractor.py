import numpy as np
from physical_feature_extractor import PhysicalFeatureExtractor

def test_hydration():
    """Test feature extraction for hydration process"""
    initial_mass = 1.0  # kg
    additive_unit_mass = 0.1  # kg
    initial_volume = 0.001  # m³
    task_type = 0  # hydration
    
    extractor = PhysicalFeatureExtractor(
        initial_mass=initial_mass,
        additive_unit_mass=additive_unit_mass,
        initial_volume=initial_volume,
        task_type=task_type
    )
    
    # Test different policies
    policies = [
        [0, 1, 1],  # low additive, medium time, medium speed
        [1, 1, 2],  # medium additive, medium time, high speed
        [2, 2, 2],  # high additive, high time, high speed
    ]
    
    # Generate dummy point clouds
    np.random.seed(42)  # For reproducibility
    initial_cloud = np.random.rand(1000, 3) * 0.01  # 1cm³ cube
    
    print("Testing hydration process:")
    print("-" * 50)
    for i, policy in enumerate(policies):
        # Simulate slightly different current clouds
        current_cloud = initial_cloud + np.random.rand(1000, 3) * 0.001
        
        features = extractor.extract_features(policy)
        print(f"\nPolicy {i+1} ({policy}):")
        print(f"Mass: {features[0]:.3f} kg")
        print(f"Density: {features[1]:.1f} kg/m³")
        print(f"E: {features[2]:.2e} Pa")
        print(f"nu: {features[3]:.3f}")
        print(f"Yield stress: {features[4]:.2e} Pa")
        print(f"Viscosity: {features[5]:.2e} Pa·s")


def test_emulsification():
    """Test feature extraction for emulsification process"""
    initial_mass = 1.0  # kg
    additive_unit_mass = 0.1  # kg
    initial_volume = 0.001  # m³
    task_type = 1  # emulsification
    
    extractor = PhysicalFeatureExtractor(
        initial_mass=initial_mass,
        additive_unit_mass=additive_unit_mass,
        initial_volume=initial_volume,
        task_type=task_type
    )
    
    # Test different policies
    policies = [
        [0, 2, 2],  # low additive, high time, high speed
        [1, 1, 1],  # medium additive, medium time, medium speed
        [2, 0, 2],  # high additive, low time, high speed
    ]
    
    # Generate dummy point clouds
    np.random.seed(42)  # For reproducibility
    initial_cloud = np.random.rand(1000, 3) * 0.01  # 1cm³ cube
    
    print("\nTesting emulsification process:")
    print("-" * 50)
    for i, policy in enumerate(policies):
        # Simulate more significant changes in point cloud for emulsification
        current_cloud = initial_cloud * (1 + 0.2 * policy[0])
        
        features = extractor.extract_features(policy)
        print(f"\nPolicy {i+1} ({policy}):")
        print(f"Mass: {features[0]:.3f} kg")
        print(f"Density: {features[1]:.1f} kg/m³")
        print(f"E: {features[2]:.2e} Pa")
        print(f"nu: {features[3]:.3f}")
        print(f"Yield stress: {features[4]:.2e} Pa")
        print(f"Viscosity: {features[5]:.2e} Pa·s")

if __name__ == "__main__":
    # Run tests
    # test_hydration()
    test_emulsification()