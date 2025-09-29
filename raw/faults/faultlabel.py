import pandas as pd
import os

def agi(file1):
    df= pd.read_csv(file1,header=0)
    df['fault_type']=0
    df.loc[(df['Time_min'] > 35) & (df['Agitator_Speed_rpm'] == 0), 'fault_type'] = 1
    fault_times = [35, 40, 55, 66, 75,45, 70, 80, 120, 175,150,180,220,260,300,37, 44, 57, 69, 85,44, 74, 81, 123, 165,140,183,228,270,250]
    window = 1  # minutes before and after
    fault_condition = False
    for t in fault_times:
        cond = (df['Time_min'] >= t - window) & (df['Time_min'] <= t + window)
        fault_condition |= cond
    df.loc[fault_condition, 'fault_type'] = 1

    return df

def runa(file2):
    df= pd.read_csv(file2,header=0)
    df['fault_type']=0
    fault_times=[46, 92, 112, 145, 198, 234, 255,260,270]
    window=2
    fault_condition = False
    for t in fault_times:
        cond = (df['Time_min'] >= t - window) & (df['Time_min'] <= t + (window+10))
        fault_condition |= cond
    df.loc[fault_condition, 'fault_type'] = 2

    fault_times1 = [85,135,160,212,280]
    fault_condit=False
    win = 2  # minutes before and after
    for t in fault_times1:
        cond1 = (df['Time_min'] >= t - win) & (df['Time_min'] <= t + (win+18))
        fault_condit |= cond1
    df.loc[fault_condit, 'fault_type'] = 2
    return df

def main():
    file1 = 'agitatorfault.csv'
    file2='runawayfault.csv'
    # Call agitator fault function
    df_agi = agi(file1)
    df_agi.to_csv('agitatorfault_L.csv', index=False)

    # Call run fault function
    df_run = runa(file2)
    df_run.to_csv('runawayfault_L.csv', index=False)
if __name__ == '__main__':
    main()
    print('datafault Labeled')





