import numpy as np
import re
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def read_sensor_data(file_path):
    pattern = re.compile(r'sensor0\s*=\s*(\d+)\s*sensor1\s*=\s*(\d+)')
    sensor0_data = []
    sensor1_data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                sensor0 = float(match.group(1))
                sensor1 = float(match.group(2))
                
                # Apply scaling
                scaled_sensor0 = (sensor0 + 317) * (1 / 9.5) 
                scaled_sensor1 = (sensor1 + 323) * (1 / 9.3)
                scaled_sensor0 *= 0.00981
                scaled_sensor1 *= 0.00981
                
                sensor0_data.append(scaled_sensor0)
                sensor1_data.append(scaled_sensor1)
    
    # Ensure both arrays have the same length
    min_length = min(len(sensor0_data), len(sensor1_data))
    return np.array(sensor0_data[:min_length]), np.array(sensor1_data[:min_length])

def remove_outliers(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return data[z_scores < threshold]

def calculate_external_force(file_path, window_size=5, polyorder=3, force_threshold=0.001):
    # Read sensor data
    sensor0_array, sensor1_array = read_sensor_data(file_path)
    
    # Ensure arrays have the same length
    min_length = min(len(sensor0_array), len(sensor1_array))
    sensor0_array = sensor0_array[:min_length]
    sensor1_array = sensor1_array[:min_length]
    
    # Remove outliers
    sensor0_array = remove_outliers(sensor0_array)
    sensor1_array = remove_outliers(sensor1_array)
    
    # Ensure arrays still have the same length after outlier removal
    min_length = min(len(sensor0_array), len(sensor1_array))
    sensor0_array = sensor0_array[:min_length]
    sensor1_array = sensor1_array[:min_length]
    
    # Calculate total gripping force
    total_force = sensor0_array + sensor1_array
    
    # Use Savitzky-Golay filter to smooth the data
    smoothed_force = savgol_filter(total_force, window_size, polyorder)
    
    # Calculate rate of force change
    force_change = np.diff(smoothed_force)
    
    # Detect significant force changes
    significant_changes = np.where(np.abs(force_change) > force_threshold)[0]
    
    # Analyze force changes
    external_force_events = []
    for i in significant_changes:
        if force_change[i] > 0:
            event = "increase"
        else:
            event = "decrease"
        external_force_events.append((i, event, force_change[i]))
    
    return smoothed_force, force_change, external_force_events

def analyze_force_data(file_path):
    # Call the function
    smoothed_force, force_change, events = calculate_external_force(file_path)

    # 1. Analyze overall force profile
    avg_force = np.mean(smoothed_force)
    max_force = np.max(smoothed_force)
    min_force = np.min(smoothed_force)

    print(f"Average gripping force: {avg_force:.2f}")
    print(f"Maximum gripping force: {max_force:.2f}")
    print(f"Minimum gripping force: {min_force:.2f}")

    # 2. Analyze force changes
    avg_change = np.mean(np.abs(force_change))
    max_increase = np.max(force_change)
    max_decrease = np.min(force_change)

    print(f"Average absolute force change: {avg_change:.2f}")
    print(f"Maximum force increase: {max_increase:.2f}")
    print(f"Maximum force decrease: {max_decrease:.2f}")

    # 3. Analyze external force events
    print("\nSignificant external force events:")
    for i, (time, event_type, magnitude) in enumerate(events, 1):
        print(f"Event {i}: {event_type.capitalize()} of {magnitude:.2f} at time {time}")

    # 4. Visualize the data
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(smoothed_force)
    plt.title('Smoothed Total Gripping Force')
    plt.ylabel('Force (units)')
    plt.axhline(y=avg_force, color='r', linestyle='--', label='Average Force')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(force_change)
    plt.title('Rate of Force Change')
    plt.ylabel('Force Change (units/time)')
    plt.xlabel('Time')

    for event in events:
        plt.axvline(x=event[0], color='g', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 5. Detect potential object slippage
    slippage_threshold = -20  # Adjust based on your system
    potential_slips = [event for event in events if event[1] == "decrease" and event[2] < slippage_threshold]
    
    if potential_slips:
        print("\nPotential object slippage detected:")
        for slip in potential_slips:
            print(f"Time: {slip[0]}, Magnitude: {slip[2]:.2f}")
    else:
        print("\nNo potential object slippage detected.")

    # 6. Estimate object weight changes
    weight_change_threshold = 15  # Adjust based on your system
    weight_changes = [event for event in events if abs(event[2]) > weight_change_threshold]
    
    if weight_changes:
        print("\nPossible object weight changes:")
        for change in weight_changes:
            direction = "increase" if change[2] > 0 else "decrease"
            print(f"Time: {change[0]}, Direction: {direction}, Magnitude: {abs(change[2]):.2f}")
    else:
        print("\nNo significant object weight changes detected.")
