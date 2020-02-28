import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split_data(KEYS):
    i = 0
    for KEY in KEYS:
        data = read_data(KEY)
        calculate_stats(data['gyro'])
        calculate_stats(data['acc'])
        inactive_timestamps = get_inactive(data)
        lis,emg_lis = get_points(data,inactive_timestamps,KEY)
        write_points(lis,emg_lis,i)
        i = i + 10

def read_data(KEY):
    # Read Data 
    gyro = pd.read_csv('./myo_keyboard_data/'+KEY+'/gyro.csv',index_col = 'timestamp')
    acc = pd.read_csv('./myo_keyboard_data/'+KEY+'/acc.csv',index_col = 'timestamp')
    orin = pd.read_csv('./myo_keyboard_data/'+KEY+'/orin.csv',index_col = 'timestamp')
    eul = pd.read_csv('./myo_keyboard_data/'+KEY+'/eul.csv',index_col = 'timestamp')
    emg = pd.read_csv('./myo_keyboard_data/'+KEY+'/emg.csv',index_col = 'timestamp')
    
    gyro.index = range(len(gyro))
    gyro.index = gyro.index*4

    acc.index = range(len(acc))
    acc.index = acc.index*4

    eul.index = range(len(eul))
    eul.index = eul.index*4

    orin.index = range(len(orin))
    orin.index = orin.index*4

    emg.index = range(len(emg))

    return  {'gyro':gyro,'acc':acc,'orin':orin,'eul':eul,'emg':emg}

def calculate_stats(df):
    # Calculate the median absolute deviation
    # and the windowed mean absolute deviation
    win_size = 7
    for col in df.columns:
        df[col + '_avg'] = df[col].mean()
        df[col+'_mad'] = df[col].mad()
        df[col+'_ad'] = (df[col] - df[col+'_avg']).abs()
        df[col+'_win_mean_ad'] = df[col].copy()
        for index in range(df[col].size):
            win_ad = df[col+'_ad'].iloc[index]
            actual = 1
            for i in range(1,win_size+1):
                if(index - i >= 0):
                    win_ad = win_ad + df[col+'_ad'].iloc[index - i]
                    actual = actual + 1
                if(index + i < df[col].size):
                    win_ad = win_ad + df[col+'_ad'].iloc[index + i]
                    actual = actual + 1
            df[col+'_win_mean_ad'].iloc[index] = win_ad/actual
    

def get_inactive(data):
    #Find all the inactive timestamps
    activity = data['gyro']['x'].copy()
    for i in range(activity.size):
        sum_win_mad = 0
        for col in ['x','y','z']:
            if(data['gyro'][col+'_win_mean_ad'].iloc[i]>data['gyro'][col+'_mad'].iloc[i]):
                sum_win_mad = 1
        for col in ['x','y','z']:
            if(data['acc'][col+'_win_mean_ad'].iloc[i]>data['acc'][col+'_mad'].iloc[i]):
                sum_win_mad = 1
        activity.iloc[i] = sum_win_mad

    return activity[activity.values == 0 ].index.values

def get_points(data,inactive_timestamps,KEY):
    #Put all data into one big dataframe
    cleaned_emg = data['emg'][['emg1','emg2','emg3','emg4',
                  'emg5','emg6','emg7','emg8']]
    cleaned_gyro = data['gyro'][['x','y','z']]
    cleaned_gyro.rename(columns = {'x':'gyro_x','y':'gyro_y','z':'gyro_z'},inplace = True)
    cleaned_orin = data['orin'][['x','y','z','w']]
    cleaned_orin.rename(columns = {'x':'orin_x','y':'orin_y','z':'orin_z','w':'orin_w'},inplace = True)
    cleaned_eul = data['eul'][['roll','pitch','yaw']]
    cleaned_acc = data['acc'][['x','y','z']]
    cleaned_acc.rename(columns = {'x':'acc_x','y':'acc_y','z':'acc_z'},inplace = True)
    beeg_df = pd.concat([cleaned_gyro,cleaned_orin,cleaned_eul,cleaned_acc],axis = 1)
    beeg_df['Key'] = KEY

    #Remove inactive timestamps and split up data
    lis = [beeg_df]
    for timestamp in inactive_timestamps:
        new_lis = []
        for i in range(len(lis)):
            curr = lis[i]
            if(not (curr.index[0] > timestamp or curr.index[-1] < timestamp)):
                before = curr.loc[:timestamp-1]
                after = curr.loc[timestamp+1:]
                if(before.size > 0):
                    new_lis.append(before)
                if(after.size > 0):
                    new_lis.append(after)
            else:
                new_lis.append(curr)
        lis = new_lis
    
    emg_lis = [cleaned_emg]
    for timestamp in inactive_timestamps:
        new_lis = []
        for i in range(len(emg_lis)):
            curr = emg_lis[i]
            if(not (curr.index[0] > timestamp or curr.index[-1] < timestamp)):
                before = curr.loc[:timestamp-1]
                after = curr.loc[timestamp+1:]
                if(before.size > 0):
                    new_lis.append(before)
                if(after.size > 0):
                    new_lis.append(after)
            else:
                new_lis.append(curr)
        emg_lis = new_lis

    # Remove all but top ten biggest windows of activity
    lis.sort(key = lambda df:-df.size)
    emg_lis.sort(key = lambda df:-df.size)
    
    lis = lis[0:10]
    emg_lis = emg_lis[0:10]
    return lis,emg_lis

def write_points(lis,emg_lis,i):
    #Write Data to folder
    j = 0
    for point in lis:
        i = i + 1
        point.to_csv('./all_phys_data_points/phys_'+str(i)+'.csv')
        emg_lis[j].to_csv('./all_emg_data_points/emg_'+str(i)+'.csv')
        j = j + 1


if __name__ == "__main__":
    KEYS = ['Right','Backward','Left','Enter','Forward']
    split_data(KEYS)
