import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_reactor_simulation_final(fault_type='none', fault_time=80, fault_magnitude=0.5):
    """
    Simulates a semibatch reactor for the nitration of benzene using
    realistic chemical stoichiometry, physical properties, and feed concentrations.
    Includes stochastic noise, a PI temperature controller, and a fixed coolant
    inlet temperature to simulate real-world process dynamics. This version
    also dynamically calculates mixture properties (density, heat capacity) and
    the heat transfer coefficient based on process conditions.

    Args:
        fault_type (str): The type of fault to introduce.
                          'none' - Normal operation.
                          'cooling_failure' - Reduces coolant flow rate.
                          'agitator_failure' - Reduces agitator speed.
                          'feed_rate_error' - Incorrect feed flow rate.
        fault_time (int): The time step (minutes) at which a fault occurs.
        fault_magnitude (float): The severity of the fault (e.g., 0.5 means 50% of normal value).

    Returns:
        pandas.DataFrame: A DataFrame containing the time-series data of the simulation.
    """
    # 1. --- Constants and Initial Conditions ---

    # Reaction properties for Benzene Nitration
    Ea = 8.3e4      # Activation Energy (J/mol)
    R = 8.314       # Gas Constant (J/mol·K)
    k0 = 2.86e8        # Pre-exponential factor (m^3/mol·s)
    deltaH = -1.35e5 # Heat of reaction (J/mol)

    # Physical Properties of individual components
    rho_coolant = 1000 # Density of coolant (water) (kg/m^3)
    Cp_coolant = 4184  # Heat capacity of coolant (water) (J/kg·K)

    # Molecular Weights (g/mol)
    MW_B = 78.11      # Benzene (C6H6)
    MW_A = 63.01      # Nitric Acid (HNO3)
    MW_H2SO4 = 98.08  # Sulfuric Acid (H2SO4) - for reference
    MW_H2O = 18.015   # Water (H2O) - for reference
    MW_NB = 123.11    # Nitrobenzene (C6H5NO2) - for reference

    # Mixed Acid Feed Composition & Properties
    acid_wt_percent_HNO3 = 24.9
    acid_density = 1680 # kg/m^3
    Cp_feed = 2760      # Heat capacity of mixed acid feed (J/kg·K)
    T_feed = 298.15     # Feed temperature (25°C)
    C_A_feed = (acid_density * (acid_wt_percent_HNO3 / 100.0) / MW_A) * 1000

    # Dynamic Property Parameters for Mixture
    rho_initial = 876    # Initial density (Benzene) (kg/m^3)
    Cp_initial = 1740    # Initial heat capacity (Benzene) (J/kg·K)
    rho_final = 1450     # Estimated final mixture density (kg/m^3)
    Cp_final = 2200      # Estimated final mixture heat capacity (J/kg·K)

    # Reactor and Benzene Load
    moles_B_initial_kmol = 100.0
    moles_B_initial = moles_B_initial_kmol * 1000
    V0 = (moles_B_initial * MW_B / 1000) / rho_initial
    C_B0 = moles_B_initial / V0

    # Feed Flow Rate and Duration
    feed_duration_min = 120
    feed_duration_s = feed_duration_min * 60
    target_mole_ratio = 1.23
    total_moles_A_to_add = moles_B_initial * target_mole_ratio
    required_feed_volume = total_moles_A_to_add / C_A_feed
    F_in = required_feed_volume / feed_duration_s

    # Equipment and Operating Parameters
    base_UA = 7900
    UA_agitator_exponent = 0.67
    T0 = 313.15
    Tc_in = 298.15 # Fixed Coolant Inlet Temperature at 25°C
    Agitator_Speed_base = 800
    Coolant_Flow_max = 1.0
    V_reactor_max = V0 + required_feed_volume + 0.5

    # Safety and Operational Clamps
    T_max = 358.15
    Agitator_Speed_max = 1200.0
    Pressure_max = 1.4

    # PI Controller parameters (Corrected for reverse-acting cooling)
    Kc = -0.8
    tau_i = 5.0
    integral_error = 0.0

    # Noise parameters
    noise_std_dev_UA = 170.0
    noise_std_dev_feed = F_in * 0.01
    noise_std_dev_coolant = Coolant_Flow_max * 0.01
    noise_std_dev_temp_sensor = 0.1

    # Simulation Time
    t_final = 250
    dt = 0.1
    n_steps = int(t_final / dt)
    time = np.linspace(0, t_final, n_steps + 1)

    UA_coeff = base_UA / (Agitator_Speed_base ** UA_agitator_exponent)

    # 2. --- Simulation Variables Setup ---
    T, T_measured, T_setpoint, Pressure, C_B, C_A, V = (np.zeros(n_steps + 1) for _ in range(7))
    Agitator_Speed = np.full(n_steps + 1, Agitator_Speed_base)
    Coolant_Flow, Tc_in_actual, Tc_out, Feed_Flow = (np.zeros(n_steps + 1) for _ in range(4))

    # Initial conditions
    T[0], C_B[0], V[0] = T0, C_B0, V0
    T_measured[0] = T0 + np.random.normal(0, noise_std_dev_temp_sensor)
    T_setpoint[0] = T0 # Start setpoint at the initial reactor temperature
    C_A[0] = 0.0
    Tc_in_actual[0], Tc_out[0] = Tc_in, Tc_in
    Pressure[0] = 1.2

    # 3. --- The Simulation Loop ---
    for i in range(1, n_steps + 1):
        current_time_min = time[i]
        dt_s = dt * 60

        # Define Temperature Setpoint Profile (Ramp from T0 to 80C)
        if current_time_min < feed_duration_min:
            T_setpoint[i] = T0 + (353.15 - T0) * (current_time_min / feed_duration_min)
        else:
            T_setpoint[i] = 353.15

        # Set Coolant Inlet Temperature (now fixed)
        Tc_in_actual[i] = Tc_in

        # PI Controller for Coolant Flow with Anti-Windup Logic
        error = T_setpoint[i] - T_measured[i-1]
        
        # Calculate the raw, unclamped controller output
        controller_output = Kc * (error + (1/tau_i) * integral_error)
        
        # Anti-windup: Only integrate if the controller is not saturated,
        # or if the error is trying to bring it out of saturation.
        is_saturated_high = (controller_output >= Coolant_Flow_max) and (error < 0) # error < 0 for reverse-acting
        is_saturated_low = (controller_output <= 0) and (error > 0) # error > 0 for reverse-acting

        if not (is_saturated_high or is_saturated_low):
            integral_error += error * dt

        # The actual setpoint for the coolant flow is the clamped controller output
        coolant_flow_setpoint = max(0, min(Coolant_Flow_max, controller_output))

        # --- Fault and Noise Injection Logic ---
        agitator_speed_setpoint = Agitator_Speed_base
        feed_flow_setpoint = F_in if current_time_min < feed_duration_min else 0.0

        if current_time_min >= fault_time:
            if fault_type == 'cooling_failure': coolant_flow_setpoint *= (1 - fault_magnitude)
            elif fault_type == 'agitator_failure': agitator_speed_setpoint *= (1 - fault_magnitude)
            elif fault_type == 'feed_rate_error' and current_time_min < feed_duration_min: feed_flow_setpoint *= (1 + fault_magnitude)

        # Final coolant flow with noise, clamped again to physical limits (0 to max)
        noisy_flow = coolant_flow_setpoint + np.random.normal(0, noise_std_dev_coolant)
        Coolant_Flow[i] = max(0, min(Coolant_Flow_max, noisy_flow))
        
        Agitator_Speed[i] = min(Agitator_Speed_max, max(0, agitator_speed_setpoint))
        Feed_Flow[i] = max(0, feed_flow_setpoint + np.random.normal(0, noise_std_dev_feed))

        # --- Dynamic Parameter Calculations ---
        feed_vol_added = V[i-1] - V0
        fraction_added = min(1.0, feed_vol_added / required_feed_volume)
        current_rho = rho_initial + (rho_final - rho_initial) * fraction_added
        current_Cp = Cp_initial + (Cp_final - Cp_initial) * fraction_added
        current_UA_from_agitator = UA_coeff * (Agitator_Speed[i] ** UA_agitator_exponent)
        current_UA = max(0, current_UA_from_agitator + np.random.normal(0, noise_std_dev_UA))

        # --- Differential Equations ---
        T_prev, V_prev, C_A_prev, C_B_prev = T[i-1], V[i-1], C_A[i-1], C_B[i-1]

        k = k0 * np.exp(-Ea / (R * T_prev))
        rate = k * C_A_prev * C_B_prev

        dC_B_dt = -rate - (Feed_Flow[i] / V_prev) * C_B_prev
        C_B[i] = C_B_prev + dC_B_dt * dt_s

        dC_A_dt = (Feed_Flow[i] / V_prev) * (C_A_feed - C_A_prev) - rate
        C_A[i] = C_A_prev + dC_A_dt * dt_s

        V[i] = V_prev + Feed_Flow[i] * dt_s

        Q_gen = -deltaH * rate * V_prev

        # --- Corrected Heat Removal Calculation ---
        flow_threshold = 1e-6 # A very small flow, effectively zero
        if Coolant_Flow[i] > flow_threshold:
            # First, calculate the coolant outlet temperature.
            Tc_out[i] = T_prev - (T_prev - Tc_in_actual[i]) * np.exp(-current_UA / (Coolant_Flow[i] * rho_coolant * Cp_coolant))
            # Then, calculate heat removed based on the energy gained by the coolant.
            # This ensures the energy balance is physically consistent.
            Q_rem = Coolant_Flow[i] * rho_coolant * Cp_coolant * (Tc_out[i] - Tc_in_actual[i])
        else:
            # If there's no coolant flow, outlet temp equals reactor temp, and no heat is removed.
            Tc_out[i] = T_prev
            Q_rem = 0.0

        # Calculate heat effect of the incoming feed.
        Q_feed = Feed_Flow[i] * acid_density * Cp_feed * (T_feed - T_prev)
        
        # Calculate the overall rate of temperature change.
        dT_dt = (Q_gen - Q_rem + Q_feed) / (current_rho * current_Cp * V_prev)

        # Update the reactor temperature.
        T[i] = min(T_prev + dT_dt * dt_s, T_max)
        T_measured[i] = T[i] + np.random.normal(0, noise_std_dev_temp_sensor)

        Pressure[i] = min(1.2 + 0.08 * (T[i] - T0), Pressure_max)
        C_A[i], C_B[i] = max(0, C_A[i]), max(0, C_B[i])

        if V[i] > V_reactor_max:
            print(f"Reactor overflow at time {current_time_min} min. Simulation stopped.")
            time = time[:i+1]
            arrays_to_truncate = [T, T_measured, T_setpoint, Pressure, C_B, C_A, V, Agitator_Speed, Coolant_Flow, Tc_in_actual, Tc_out, Feed_Flow]
            truncated_arrays = [arr[:i+1] for arr in arrays_to_truncate]
            T, T_measured, T_setpoint, Pressure, C_B, C_A, V, Agitator_Speed, Coolant_Flow, Tc_in_actual, Tc_out, Feed_Flow = truncated_arrays
            break

    # 4. --- Saving the Results ---
    results = pd.DataFrame({
        'Time_min': time, 'Temperature_K': T, 'Temperature_Measured_K': T_measured,
        'Temperature_Setpoint_K': T_setpoint, 'Pressure_bar': Pressure,
        'Benzene_Concentration_mol_m3': C_B, 'NitricAcid_Concentration_mol_m3': C_A,
        'Volume_m3': V, 'Agitator_Speed_rpm': Agitator_Speed[:len(time)],
        'Coolant_Flow_m3_s': Coolant_Flow[:len(time)], 'Coolant_In_Temp_K': Tc_in_actual[:len(time)],
        'Coolant_Out_Temp_K': Tc_out[:len(time)], 'Feed_Flow_m3_s': Feed_Flow[:len(time)]
    })
    return results

if __name__ == '__main__':
    normal_data = run_reactor_simulation_final()
    normal_data.to_csv('simdata/normal_operation_nitration_dynamic6.csv', index=False)
    print("Dynamic simulation complete. Data saved to 'normal_operation_nitration_dynamic.csv'")

    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Semibatch Benzene Nitration Simulation (Dynamic Parameters)', fontsize=16)

    # Temperature Plot
    axs[0, 0].plot(normal_data['Time_min'], normal_data['Temperature_Setpoint_K'] - 273.15, label='Setpoint', color='green', linestyle=':')
    axs[0, 0].plot(normal_data['Time_min'], normal_data['Temperature_Measured_K'] - 273.15, label='Measured', color='blue', alpha=0.6)
    axs[0, 0].plot(normal_data['Time_min'], normal_data['Temperature_K'] - 273.15, label='Actual', color='black', linestyle='--')
    axs[0, 0].set_title('Reactor Temperature'); axs[0, 0].set_ylabel('Temperature (°C)'); axs[0, 0].legend(); axs[0, 0].grid(True)

    # Concentration Plot
    axs[0, 1].plot(normal_data['Time_min'], normal_data['Benzene_Concentration_mol_m3'], label='Benzene')
    axs[0, 1].plot(normal_data['Time_min'], normal_data['NitricAcid_Concentration_mol_m3'], label='Nitric Acid')
    axs[0, 1].set_title('Reactant Concentrations'); axs[0, 1].set_ylabel('Concentration (mol/m³)'); axs[0, 1].legend(); axs[0, 1].grid(True)

    # Volume Plot
    axs[1, 0].plot(normal_data['Time_min'], normal_data['Volume_m3'])
    axs[1, 0].set_title('Reactor Volume'); axs[1, 0].set_ylabel('Volume (m³)'); axs[1, 0].grid(True)

    # Pressure Plot
    axs[1, 1].plot(normal_data['Time_min'], normal_data['Pressure_bar'])
    axs[1, 1].set_title('Reactor Pressure'); axs[1, 1].set_ylabel('Pressure (bar)'); axs[1, 1].grid(True)

    # Agitator Speed
    axs[2, 0].plot(normal_data['Time_min'], normal_data['Agitator_Speed_rpm'])
    axs[2, 0].set_title('Agitator Speed'); axs[2, 0].set_ylabel('Speed (rpm)'); axs[2, 0].grid(True)

    # Coolant Flow
    axs[2, 1].plot(normal_data['Time_min'], normal_data['Coolant_Flow_m3_s'])
    axs[2, 1].set_title('Coolant Flow'); axs[2, 1].set_ylabel('Flow (m³/s)'); axs[2, 1].grid(True)

    # Feed Flow
    axs[3, 0].plot(normal_data['Time_min'], normal_data['Feed_Flow_m3_s'])
    axs[3, 0].set_title('Acid Feed Flow'); axs[3, 0].set_ylabel('Flow (m³/s)'); axs[3, 0].grid(True)

    # Coolant Temperature
    ax31_2 = axs[3, 1].twinx()
    axs[3, 1].plot(normal_data['Time_min'], normal_data['Coolant_Out_Temp_K'] - 273.15, label='Out', color='red')
    ax31_2.plot(normal_data['Time_min'], normal_data['Coolant_In_Temp_K'] - 273.15, label='In', color='blue', linestyle='--')
    axs[3, 1].set_title('Coolant Temperatures'); axs[3, 1].set_ylabel('Outlet Temp (°C)', color='red'); ax31_2.set_ylabel('Inlet Temp (°C)', color='blue'); axs[3, 1].grid(True)

    for ax in axs.flat:
        ax.set_xlabel('Time (min)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

