import pandas as pd
import numpy as np

def add_fluctuations(df, column='ReactorTemp', mean=2, std_dev=0.4):
    noise = np.random.normal(loc=mean, scale=std_dev, size=len(df))
    df[column] = df[column] + noise
    return df

def agifault(filepath):
    df=pd.read_csv(filepath,header=0)
    df.loc[df['Time_min'] > 35, 'Agitator_Speed_rpm'] = 0
    df.loc[df['Time_min'].isin([35, 40, 55, 66, 75]), 'ReactorTemp'] = 340
    df.loc[df['Time_min'].isin([45, 70, 80, 120, 175,150,180,220,260,300]), 'ReactorTemp'] = 330
    df.loc[df['Time_min'].isin([37, 44, 57, 69, 85]), 'ReactorTemp'] = 380
    df.loc[df['Time_min'].isin([44, 74, 81, 123, 165,140,183,228,270,250]), 'ReactorTemp'] = 375
    df['ReactorTemp'] = df['ReactorTemp'].ewm(span=5, adjust=False).mean()
    df.loc[df['Time_min'] > 100, 'ReactorTemp'] = df.loc[df['Time_min'] > 100, 'ReactorTemp'] + 18
    df = add_fluctuations(df, 'ReactorTemp', std_dev=0.5)
    df.loc[df['Time_min'].isin([35, 40, 55, 66, 75]), 'Pressure_bar'] = 1.35
    df.loc[df['Time_min'].isin([45, 70, 80, 120, 175,150,180,220,260,300]), 'Pressure_bar'] = 1.28
    df.loc[df['Time_min'].isin([37, 44, 57, 69, 85]), 'Pressure_bar'] = 1.72
    df.loc[df['Time_min'].isin([44, 74, 81, 123, 165,140,183,228,270,250]), 'Pressure_bar'] = 1.6
    df['Pressure_bar'] = df['Pressure_bar'].ewm(span=2, adjust=False).mean()
    df.loc[df['Time_min'] > 35, 'Pressure_bar'] = df.loc[df['Time_min'] > 35, 'Pressure_bar'] + 0.175
      
    df.loc[df['Time_min'] > 80, 'Coolant_Flow_m3_s'] = df.loc[df['Time_min'] > 80, 'Coolant_Flow_m3_s'] + 0.54
    df.loc[df['Time_min'] > 35, 'Coolant_delatT_K'] = df.loc[df['Time_min'] > 35, 'Coolant_delatT_K'] + 5
    df.loc[df['Time_min'] > 100, 'Coolant_delatT_K'] = df.loc[df['Time_min'] > 100, 'Coolant_delatT_K'] + 3
    df.loc[df['Time_min'].isin([35, 40, 55, 66, 75]), 'Coolant_delatT_K'] = 47
    df.loc[df['Time_min'].isin([45, 70, 80, 120, 175,150,180,220,260,300]), 'Coolant_delatT_K'] = 46
    df.loc[df['Time_min'].isin([37, 44, 57, 69, 85]), 'Coolant_delatT_K'] = 45
    df.loc[df['Time_min'].isin([44, 74, 81, 123, 165,140,183,228,270,250]), 'Coolant_delatT_K'] = 48
    df = add_fluctuations(df, 'Coolant_delatT_K', std_dev=0.3)

    return df
def runfault(filepath):
    df=pd.read_csv(filepath,header=0)
    max_ramp_length = 60  # number of points over which to ramp
    max_values = [370, 380, 390, 404, 410, 421, 432, 450]
    min_values = [340, 350, 358, 360, 368, 374, 369, 381]
    start_times = [46, 92, 112, 145, 198, 234, 255, 271]

    for start_time, min_val, max_val in zip(start_times, min_values, max_values):
        ramp_values = np.linspace(min_val, max_val, max_ramp_length)
        start_idx = df.index[df['Time_min'] == start_time][0]
        end_idx = start_idx + max_ramp_length - 1
        
        # Ensure the range is within DataFrame length
        if end_idx >= len(df):
            end_idx = len(df) - 1
            ramp_values = ramp_values[:end_idx - start_idx + 1]
        
        df.loc[start_idx:end_idx, 'ReactorTemp'] = ramp_values
    df.loc[(df['Time_min'] > 50) & (df['ReactorTemp'] < 350), 'ReactorTemp'] = 367
    df['ReactorTemp'] = df['ReactorTemp'].ewm(span=3, adjust=False).mean()
    df = add_fluctuations(df, 'ReactorTemp', std_dev=0.4)
    max_ramp_length = 60  # number of points over which to ramp
    max_values = [1.5, 1.65, 1.7, 1.77, 1.83, 1.98, 2.2, 2.4]
    min_values = [1.4, 1.48, 1.53, 1.58, 1.64, 1.7, 1.8, 1.9]
    start_times = [46, 92, 112, 145, 198, 234, 255, 271]

    for start_time, min_val, max_val in zip(start_times, min_values, max_values):
        ramp_values = np.linspace(min_val, max_val, max_ramp_length)
        start_idx = df.index[df['Time_min'] == start_time][0]
        end_idx = start_idx + max_ramp_length - 1
        
        # Ensure the range is within DataFrame length
        if end_idx >= len(df):
            end_idx = len(df) - 1
            ramp_values = ramp_values[:end_idx - start_idx + 1]
        
        df.loc[start_idx:end_idx, 'Pressure_bar'] = ramp_values
    df['Pressure_bar'] = df['Pressure_bar'].ewm(span=3, adjust=False).mean()
    df.loc[df['Time_min'] > 77, 'Coolant_Flow_m3_s'] = df.loc[df['Time_min'] > 77, 'Coolant_Flow_m3_s'] + 0.97
    max_ramp_length = 60  # number of points over which to ramp
    min_values = [0.0022, 0.0018, 0.0012, 0.0004 ]
    max_values = [0.0025, 0.002, 0.0015, 0.0008]
    start_times = [46, 92, 112, 145]

    for start_time, min_val, max_val in zip(start_times, min_values, max_values):
        ramp_values = np.linspace(min_val, max_val, max_ramp_length)
        start_idx = df.index[df['Time_min'] == start_time][0]
        end_idx = start_idx + max_ramp_length - 1
        # Ensure the range is within DataFrame length
        if end_idx >= len(df):
            end_idx = len(df) - 1
            ramp_values = ramp_values[:end_idx - start_idx + 1]
        
        df.loc[start_idx:end_idx, 'Feed_Flow_m3_s'] = ramp_values

    df['Feed_Flow_m3_s'] = df['Feed_Flow_m3_s'].ewm(span=3, adjust=False).mean()
    df.loc[df['Time_min'] > 85, 'Coolant_delatT_K'] = df.loc[df['Time_min'] > 85, 'Coolant_delatT_K'] + 15
    df.loc[df['Time_min'] > 135, 'Coolant_delatT_K'] = df.loc[df['Time_min'] > 135, 'Coolant_delatT_K'] + 10
    df.loc[df['Time_min'] > 160, 'Coolant_delatT_K'] = df.loc[df['Time_min'] > 160, 'Coolant_delatT_K'] + 10
    df.loc[df['Time_min'] > 212, 'Coolant_delatT_K'] = df.loc[df['Time_min'] > 212, 'Coolant_delatT_K'] + 12
    df.loc[df['Time_min'] > 253, 'Coolant_delatT_K'] = df.loc[df['Time_min'] > 253, 'Coolant_delatT_K'] + 6
    df['Feed_Flow_m3_s'] = df['Feed_Flow_m3_s'].ewm(span=3, adjust=False).mean()
    return df

def main():
    filepath = '../raw/normal/normalbatch1.csv'  # Replace with your actual CSV file path
    
    # Call agitator fault function
    df_agi = agifault(filepath)
    df_agi.to_csv('../raw/faults/agitatorfault.csv', index=False)

    # Call run fault function
    df_run = runfault(filepath)
    df_run.to_csv('../raw/faults/runawayfault.csv', index=False)

if __name__ == '__main__':
    main()
    print('datafault Generated')









