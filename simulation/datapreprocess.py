import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(filepath,values):
    df=pd.read_csv(filepath,header= 0)
    df=df.drop(['Temperature_K','Temperature_Setpoint_K','AceticAcid_Concentration_mol_m3','Ethanol_Concentration_mol_m3'],axis=1)
    df= df.rename(columns={'Temperature_Measured_K':'ReactorTemp'})
    df = df.iloc[::5, :].reset_index(drop=True)
    # Filter dataframe for Time_min <= 78
    df_segment = df[df['Time_min'] <= 68]

    # Calculate how many times to repeat the values to cover df_segment length
    repeat_times = -(-len(df_segment) // len(values))  # Ceiling division

    # Repeat the values
    repeated_values = np.tile(values, repeat_times)[:len(df_segment)]

    # Assign these repeated values back to df_segment in original DataFrame
    df.loc[df_segment.index, 'Coolant_Out_Temp_K'] = repeated_values
    df['Smoothed_Temp'] = df['Coolant_Out_Temp_K'].ewm(span=20, adjust=False).mean()
    df=df.drop(['Coolant_Out_Temp_K'],axis=1)
    df=df.rename(columns={'Smoothed_Temp':'Coolant_Out_Temp_K'})
    #df.loc[df['Feed_Flow_m3_s'] < 0.002, 'Feed_Flow_m3_s'] = 0
    df['Coolant_delatT_K']= df['Coolant_Out_Temp_K']-df['Coolant_In_Temp_K']
    df=df.drop(['Coolant_Out_Temp_K','Coolant_In_Temp_K'],axis=1)

    return df

if __name__ == '__main__':
    values = np.array([
    303.2271432, 337.9983135, 337.9455438, 337.893198, 337.8395896, 303.1880651,
    318.4998884, 337.6198867, 337.5673981, 337.5171879, 337.4647304, 334.8095336,
    318.3589687, 337.2704717, 337.2180799, 337.1635057, 314.8811613, 308.9680478,
    336.9545027, 336.9062543, 324.4951693, 323.815591, 336.7113824, 336.6614539,
    336.6103471, 316.0131147, 336.4853655, 336.4346755, 336.3853444, 336.3369186,
    336.2875021, 336.2390678, 324.4782052, 336.1221035, 336.073106, 336.0234377,
    317.4586577
    ])
    filepath='simdata/normal_operation_esterification2.csv'
    normal_data = preprocess(filepath,values)
    normal_data.to_csv('../raw/Transferlearning/normalbatcht.csv', index=False)
    print("Data saved to 'raw/normal/batch.csv'")

