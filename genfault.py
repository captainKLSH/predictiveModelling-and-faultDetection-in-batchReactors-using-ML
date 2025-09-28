import pandas as pd
import matplotlib.pyplot as plt
from reactor_simulation import run_reactor_simulation_final

def generate_and_plot_fault_data():
    """
    Runs the reactor simulation for normal operation and various fault conditions,
    saves the data to CSV files, and generates a comparative plot.
    """
    # --- 1. Define Fault Scenarios ---
    fault_scenarios = [
        {'type': 'cooling_failure', 'magnitude': 0.5, 'label': 'Cooling Failure (50%)'},
        {'type': 'agitator_failure', 'magnitude': 0.5, 'label': 'Agitator Failure (50%)'},
        {'type': 'feed_rate_error', 'magnitude': 0.5, 'label': 'Feed Rate Error (+50%)'}
    ]
    
    fault_time = 30 # Time in minutes when the fault occurs

    # --- 2. Run Normal Operation Simulation ---
    print("Running simulation for normal operation...")
    normal_data = run_reactor_simulation_final()
    normal_data.to_csv('normal_operation_data.csv', index=False)
    print("Saved normal_operation_data.csv")

    # --- 3. Run Fault Simulations ---
    fault_results = {}
    for scenario in fault_scenarios:
        print(f"Running simulation for: {scenario['label']}...")
        fault_data = run_reactor_simulation_final(
            fault_type=scenario['type'],
            fault_time=fault_time,
            fault_magnitude=scenario['magnitude']
        )
        # Save the data to a unique CSV file
        filename = f"{scenario['type']}_data.csv"
        fault_data.to_csv(filename, index=False)
        print(f"Saved {filename}")
        fault_results[scenario['label']] = fault_data

    # --- 4. Generate Comparative Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    # Plot Normal Operation
    plt.plot(normal_data['Time_min'], normal_data['Temperature_K'] - 273.15, 
             label='Normal Operation', color='blue', linewidth=2)

    # Plot Fault Conditions
    colors = {'Cooling': 'red', 'Agitator': 'orange', 'Feed': 'purple'}
    for label, data in fault_results.items():
        key = label.split(' ')[0]
        plt.plot(data['Time_min'], data['Temperature_K'] - 273.15, 
                 label=label, color=colors.get(key, 'gray'), linestyle='--')

    # Add a vertical line to show when the fault occurs
    plt.axvline(x=fault_time, color='black', linestyle=':', linewidth=2, label=f'Fault introduced at {fault_time} min')

    plt.title('Reactor Temperature Under Normal vs. Fault Conditions', fontsize=16)
    plt.xlabel('Time (min)', fontsize=12)
    plt.ylabel('Reactor Temperature (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    generate_and_plot_fault_data()
