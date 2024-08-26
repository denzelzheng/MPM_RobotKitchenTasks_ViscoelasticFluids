import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.integrate import odeint
import os
from scipy.signal import savgol_filter

# Experimental data
depths = np.arange(0.1, 1.6, 0.1) * 1e-3  # From 0.1mm to 1.5mm, step 0.1mm

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

force_data = current_directory + '/force_data_30.npy'
forces = np.load(force_data)

# Time data (assumed to be 0s, 2.5s, 5s, 7.5s, 10s)
times = np.array([0., 2.5, 5, 7.5, 10]) + 0.5

# Sample and indenter geometric parameters
sample_side = 0.04   # Sample side length, unit: m (4cm)
indenter_radius = 0.015  # Indenter radius, unit: m (diameter 3cm)

def viscoelastoplastic_model(y, t, params, strain):
    """
    Improved viscoelastoplastic constitutive model, maintaining original structure and improving viscosity calculation
    y[0]: Plastic strain
    y[1]: Viscous strain
    y[2]: Elastic strain 1 (FE-FP branch)
    y[3]: Elastic strain 2 (FE-FV branch)
    """
    E, nu, yield_stress, viscosity = params
    mu = E / (2 * (1 + nu))  # Shear modulus
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
    
    dydt = [0, 0, 0, 0]
    
    # FE-FP branch
    elastic_strain_1 = strain - y[0] - y[2]
    stress_1 = E * elastic_strain_1
    
    # Simplified plastic model (perfect plasticity, no hardening)
    f = np.abs(stress_1) - yield_stress
    if f > 0:
        dydt[0] = f / E * np.sign(stress_1)
    dydt[2] = strain - y[0] - y[2] - elastic_strain_1  # Update elastic strain 1
    
    # FE-FV branch
    elastic_strain_2 = strain - y[1] - y[3]
    
    # Improved viscous behavior calculation
    alpha = 2.0 * mu / viscosity
    beta = 2.0 * (2.0 * mu + 3 * lam) / (9.0 * viscosity) - 2.0 * mu / (viscosity * 3)
    
    A = 1 / (1 + alpha * t)
    B = t * beta / (1 + t * (alpha + 3 * beta))
    
    epsilon = elastic_strain_2
    epsilon_trace = epsilon  # For uniaxial case, trace is the strain itself
    
    temp_epsilon = A * (epsilon - B * epsilon_trace)
    viscous_stress = (2 * mu + lam) * (np.exp(temp_epsilon) - 1)
    
    dydt[1] = viscous_stress / viscosity
    dydt[3] = strain - y[1] - y[3] - elastic_strain_2  # Update elastic strain 2
    
    return dydt

def calculate_stress(strain, y, params):
    """
    Calculate total stress
    """
    E = params[0]
    
    # Stress from FE-FP branch
    stress_1 = E * (strain - y[0] - y[2])
    
    # Stress from FE-FV branch
    stress_2 = E * (strain - y[1] - y[3])
    
    # Total stress is the sum of stresses from both branches
    total_stress = stress_1 + stress_2
    
    return total_stress

def stress_function(t, E, nu, yield_stress, viscosity, strain):
    """
    Solve the viscoelastoplastic model
    """
    params = [E, nu, yield_stress, viscosity]
    y0 = [0, 0, 0, 0]  # Initial conditions: plastic strain, viscous strain, elastic strain 1, elastic strain 2 all zero
    sol = odeint(viscoelastoplastic_model, y0, t, args=(params, strain))
    stress = [calculate_stress(strain, y, params) for y in sol]
    return np.array(stress)


def calculate_yield_stress(strains, stresses, window_size=21, poly_order=3, prominence_factor=0.5):
    """
    Calculate yield stress using second derivative method, looking for a more pronounced yield plateau.
    If the yield plateau is not obvious, return a larger stress value.
    """
    window_size = min(window_size, len(strains) - 2)
    window_size = window_size if window_size % 2 == 1 else window_size - 1
    poly_order = min(poly_order, window_size - 1)
    
    smooth_stress = savgol_filter(stresses, window_size, poly_order)
    first_derivative = np.gradient(smooth_stress, strains)
    second_derivative = np.gradient(first_derivative, strains)
    
    peaks = []
    for i in range(1, len(second_derivative) - 1):
        if second_derivative[i-1] < second_derivative[i] > second_derivative[i+1]:
            peaks.append((i, second_derivative[i]))
    
    peaks.sort(key=lambda x: x[1], reverse=True)
    significant_peaks = [p for p in peaks if p[1] > peaks[0][1] * prominence_factor]
    
    if significant_peaks:
        yield_index = significant_peaks[0][0]
    else:
        # If no significant peak, return 80% of the maximum stress as yield stress
        max_stress = np.max(stresses)
        yield_stress = 0.8 * max_stress
        yield_strain = strains[np.argmin(np.abs(stresses - yield_stress))]
        return yield_stress, yield_strain
    
    yield_stress = stresses[yield_index]
    yield_strain = strains[yield_index]
    
    def piecewise_linear(x, x0, y0, k1, k2):
        return np.where(x < x0, k1 * x + y0 - k1 * x0, k2 * (x - x0) + y0)
    
    p0 = [yield_strain, yield_stress, yield_stress/yield_strain, 0]
    try:
        popt, _ = curve_fit(piecewise_linear, strains, stresses, p0)
        yield_strain = popt[0]
        yield_stress = popt[1]
    except:
        pass
    
    # If the calculated yield stress is less than 50% of the maximum stress, return 80% of the maximum stress as yield stress
    max_stress = np.max(stresses)
    if yield_stress < 0.5 * max_stress:
        yield_stress = 0.95 * max_stress
        yield_strain = strains[np.argmin(np.abs(stresses - yield_stress))]
    
    return yield_stress, yield_strain

def calculate_initial_elastic_modulus(strains, stresses, yield_strain):
    """
    Calculate initial elastic modulus
    """
    # Only use data before yield point
    mask = strains <= yield_strain
    initial_strains = strains[mask]
    initial_stresses = stresses[mask]
    
    # Use linear regression to calculate initial slope
    coeffs = np.polyfit(initial_strains, initial_stresses, 1)
    initial_E = coeffs[0]
    
    return initial_E

def fit_model(depths, forces, times, initial_yield_stress, initial_E):
    """
    Fit model to experimental data, using multiple attempts and error handling, and optimize yield stress and E
    """
    strains = depths / sample_side
    initial_params_list = [
        [initial_E, 0.3, 1e5, initial_yield_stress],
        [initial_E * 1.1, 0.4, 1e6, initial_yield_stress * 1.1],
        [initial_E * 0.9, 0.2, 1e4, initial_yield_stress * 0.9]
    ]
    
    all_params = []
    for i, strain in enumerate(strains):
        best_params = None
        min_error = np.inf
        
        for initial_params in initial_params_list:
            try:
                params, _ = curve_fit(
                    lambda t, E, nu, eta, sigma_y: stress_function(t, E, nu, sigma_y, eta, strain),
                    times, forces[i], p0=initial_params, 
                    bounds=([initial_E*0.5, 0.1, 1e3, initial_yield_stress*0.5], 
                            [initial_E*1.5, 0.49, 1e8, initial_yield_stress*1.5]),
                    method='trf', maxfev=5000
                )
                
                fitted_forces = stress_function(times, *params, strain)
                error = np.mean((forces[i] - fitted_forces)**2)
                
                if error < min_error:
                    min_error = error
                    best_params = params
            
            except RuntimeError as e:
                print(f"Fitting failed at depth {depths[i]*1000:.1f}mm: {str(e)}")
        
        if best_params is not None:
            all_params.append(best_params)
        else:
            print(f"All attempts failed at depth {depths[i]*1000:.1f}mm, using initial parameters")
            all_params.append(initial_params_list[0])
    
    return np.array(all_params)

def calculate_material_properties(depths, forces, fitted_params):
    """
    Calculate material properties
    """
    strains = depths / sample_side
    stresses = forces[:, 0] / (np.pi * indenter_radius**2)
    
    elastic_moduli = fitted_params[:, 0]
    poisson_ratios = fitted_params[:, 1]
    viscosities = fitted_params[:, 2]
    yield_stresses = fitted_params[:, 3]
    
    avg_elastic_modulus = np.mean(elastic_moduli)
    avg_poisson_ratio = np.mean(poisson_ratios)
    avg_viscosity = np.mean(viscosities)
    avg_yield_stress = np.mean(yield_stresses)
    
    return avg_elastic_modulus, elastic_moduli, avg_yield_stress, yield_stresses, avg_poisson_ratio, avg_viscosity

def plot_results(depths, forces, times, fitted_params, avg_elastic_modulus, elastic_moduli, avg_yield_stress, yield_stresses):
    """
    Plot experimental data and fitting results, including relaxation curves for all depths
    """
    num_depths = len(depths)
    num_rows = (num_depths + 3) // 2  # Calculate number of rows needed
    
    plt.figure(figsize=(20, 10 * num_rows))
    
    # Force-displacement curve
    plt.subplot(num_rows, 2, 1)
    plt.plot(depths * 1000, [f[0] for f in forces], 'o-')
    plt.title('Force-Displacement Curve')
    plt.xlabel('Depth (mm)')
    plt.ylabel('Initial Force (N)')
    
    # Stress-strain curve
    plt.subplot(num_rows, 2, 2)
    strains = depths / sample_side
    stresses = [f[0] / (np.pi * indenter_radius**2) for f in forces]
    plt.plot(strains, stresses, 'o-', label='Experimental Data')
    plt.plot(strains, avg_elastic_modulus * strains * 2, '--', label='Average Elastic Response')
    plt.axhline(y=avg_yield_stress, color='r', linestyle='--', label='Average Yield Stress')
    plt.title('Stress-Strain Curve')
    plt.xlabel('Strain')
    plt.ylabel('Stress (Pa)')
    plt.legend()
    
    # Relaxation curves for all depths
    for i, depth in enumerate(depths):
        plt.subplot(num_rows, 2, i + 3)
        plt.plot(times, forces[i], 'o', label='Experimental Data')
        
        strain = depth / sample_side
        E, nu, eta, sigma_y = fitted_params[i]
        smooth_times = np.linspace(0, 10, 100)
        fitted_forces = stress_function(smooth_times, E, nu, sigma_y, eta, strain) * (0.015 ** 2 * np.pi)
        plt.plot(smooth_times, fitted_forces, '-', label='Fitted Model')
        
        plt.title(f'Relaxation Curve at {depth*1000:.1f}mm')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
    
    # Local elastic modulus and yield stress variation with depth
    plt.subplot(num_rows, 2, num_rows * 2)
    plt.plot(depths * 1000, elastic_moduli, 'o-', label='Local Elastic Modulus')
    plt.plot(depths * 1000, yield_stresses, 's-', label='Local Yield Stress')
    plt.axhline(y=avg_elastic_modulus, color='r', linestyle='--', label='Average Elastic Modulus')
    plt.axhline(y=avg_yield_stress, color='g', linestyle='--', label='Average Yield Stress')
    plt.title('Local Elastic Modulus and Yield Stress vs Depth')
    plt.xlabel('Depth (mm)')
    plt.ylabel('Modulus / Stress (Pa)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # First calculate initial yield stress and initial elastic modulus
    strains = depths / sample_side
    stresses = forces[:, 0] / (np.pi * indenter_radius**2)
    initial_yield_stress, yield_strain = calculate_yield_stress(strains, stresses)
    initial_E = calculate_initial_elastic_modulus(strains, stresses, yield_strain)
    
    # Fit model
    fitted_params = fit_model(depths, forces, times, initial_yield_stress, initial_E)
    
    # Calculate material properties
    avg_elastic_modulus, elastic_moduli, avg_yield_stress, yield_stresses, avg_poisson_ratio, avg_viscosity = calculate_material_properties(depths, forces, fitted_params)
        
    # Print results
    print(f"Initial Elastic Modulus: {initial_E:.2e} Pa")
    print(f"Average Elastic Modulus: {avg_elastic_modulus:.2e} Pa")
    print(f"Average Poisson's Ratio: {avg_poisson_ratio:.2f}")
    print(f"Initial Yield Stress: {initial_yield_stress:.2e} Pa")
    print(f"Average Yield Stress: {avg_yield_stress:.2e} Pa")
    print(f"Average Viscosity: {avg_viscosity:.2e} Pa·s")

    # Calculate relaxation time
    relaxation_time = avg_viscosity / avg_elastic_modulus
    print(f"Relaxation Time: {relaxation_time:.2e} s")
    # Plot results
    plot_results(depths, forces, times, fitted_params, avg_elastic_modulus, elastic_moduli, avg_yield_stress, yield_stresses)

if __name__ == "__main__":
    main()