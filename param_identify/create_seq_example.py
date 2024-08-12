import numpy as np
import os

def create_smooth_sequence(steps, dimensions):
    """Create a smooth sequence of vectors."""
    t = np.linspace(0, 1, steps)
    sequence = np.zeros((steps, dimensions))
    
    for d in range(dimensions):
        # Use different frequencies for each dimension to create variety
        frequency = 1 + d
        phase = np.random.rand() * 2 * np.pi
        amplitude = 0.1  # To ensure values are between 0 and 1
        offset = 0.1  # To center the oscillation around 0.5
        
        sequence[:, d] = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
    
    return sequence

def create_force_sequence(steps):
    """Create a smooth sequence of scalar forces."""
    t = np.linspace(0, 1, steps)
    frequency = 2
    phase = np.random.rand() * 2 * np.pi
    amplitude = 0.1
    offset = 0.1
    
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

def main():
    # Set parameters
    steps = 2000
    dimensions = 3  # 3D trajectory
    
    # Generate data
    trajectory_data = create_smooth_sequence(steps, dimensions)
    force_data = create_force_sequence(steps)
    
    # Get current directory
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    
    # Save data
    traj_path = os.path.join(current_directory, "tool_traj.npy")
    force_path = os.path.join(current_directory, "tool_force.npy")
    
    np.save(traj_path, trajectory_data)
    np.save(force_path, force_data)
    
    print(f"Trajectory data saved to: {traj_path}")
    print(f"Force data saved to: {force_path}")
    print(f"Trajectory shape: {trajectory_data.shape}")
    print(f"Force shape: {force_data.shape}")
    print(f"Trajectory range: [{trajectory_data.min():.2f}, {trajectory_data.max():.2f}]")
    print(f"Force range: [{force_data.min():.2f}, {force_data.max():.2f}]")

if __name__ == "__main__":
    main()