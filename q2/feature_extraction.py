import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_feature(point,emg,phys,emg_bool):
    feature = {}  
    # get median
    if(phys):
        feature['acc_x_median'] = point['acc_x'].median()
        feature['acc_y_median'] = point['acc_y'].median()
        feature['acc_z_median'] = point['acc_z'].median()

        feature['gyro_x_median'] = point['gyro_x'].median()
        feature['gyro_y_median'] = point['gyro_y'].median()
        feature['gyro_z_median'] = point['gyro_z'].median()

        feature['orin_x_median'] = point['orin_x'].median()
        feature['orin_y_median'] = point['orin_y'].median()
        feature['orin_z_median'] = point['orin_z'].median()
        feature['orin_w_median'] = point['orin_w'].median()

        feature['eul_roll_median'] = point['roll'].median()
        feature['eul_pitch_median'] = point['pitch'].median()
        feature['eul_yaw_median'] = point['yaw'].median()

        # get median absolute deviaton
        feature['acc_x_mad'] = point['acc_x'].mad()
        feature['acc_y_mad'] = point['acc_y'].mad()
        feature['acc_z_mad'] = point['acc_z'].mad()

        feature['gyro_x_mad'] = point['gyro_x'].mad()
        feature['gyro_y_mad'] = point['gyro_y'].mad()
        feature['gyro_z_mad'] = point['gyro_z'].mad()

        feature['orin_x_mad'] = point['orin_x'].mad()
        feature['orin_y_mad'] = point['orin_y'].mad()
        feature['orin_z_mad'] = point['orin_z'].mad()
        feature['orin_w_mad'] = point['orin_w'].mad()

        feature['eul_roll_mad'] = point['roll'].mad()
        feature['eul_pitch_mad'] = point['pitch'].mad()
        feature['eul_yaw_mad'] = point['yaw'].mad()
        
        # get magnitude
        feature['gyro_mag'] = (point['gyro_x'].pow(2) + point['gyro_y'].pow(2) + point['gyro_z'].pow(2)).pow(.5).mean()
        feature['acc_mag'] = (point['acc_x'].pow(2) + point['acc_y'].pow(2) + point['acc_z'].pow(2)).pow(.5).mean()
        feature['orin_mag'] = (point['orin_x'].pow(2) + point['orin_y'].pow(2)
                          + point['orin_z'].pow(2) + point['orin_w'].pow(2)).pow(.5).mean()
        feature['eul_mag'] = (point['pitch'].pow(2) + point['yaw'].pow(2) + point['roll'].pow(2)).pow(.5).mean()

    if(emg_bool): 
        feature['emg1_var'] = emg['emg1'].var()
        feature['emg2_var'] = emg['emg2'].var()
        feature['emg3_var'] = emg['emg3'].var()
        feature['emg4_var'] = emg['emg4'].var()
        feature['emg5_var'] = emg['emg5'].var()
        feature['emg6_var'] = emg['emg6'].var()
        feature['emg7_var'] = emg['emg7'].var()
        feature['emg8_var'] = emg['emg8'].var()

        feature['emg1_mav'] = emg['emg1'].abs().mean()
        feature['emg2_mav'] = emg['emg2'].abs().mean()
        feature['emg3_mav'] = emg['emg3'].abs().mean()
        feature['emg4_mav'] = emg['emg4'].abs().mean()
        feature['emg5_mav'] = emg['emg5'].abs().mean()
        feature['emg6_mav'] = emg['emg6'].abs().mean()
        feature['emg7_mav'] = emg['emg7'].abs().mean()
        feature['emg8_mav'] = emg['emg8'].abs().mean()
    
        feature['emg1_wl'] = (emg['emg1'].reset_index(drop = True) - emg['emg1'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
        feature['emg2_wl'] = (emg['emg2'].reset_index(drop = True) - emg['emg2'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
        feature['emg3_wl'] = (emg['emg3'].reset_index(drop = True) - emg['emg3'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
        feature['emg4_wl'] = (emg['emg4'].reset_index(drop = True) - emg['emg4'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
        feature['emg5_wl'] = (emg['emg5'].reset_index(drop = True) - emg['emg5'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
        feature['emg6_wl'] = (emg['emg6'].reset_index(drop = True) - emg['emg6'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
        feature['emg7_wl'] = (emg['emg7'].reset_index(drop = True) - emg['emg7'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
        feature['emg8_wl'] = (emg['emg8'].reset_index(drop = True) - emg['emg8'].iloc[1:].reset_index(drop = True)).iloc[0:-1].sum()
    

    feature['label'] = point['Key'].iloc[0]
    
    return feature

if __name__ == "__main__":
    point = pd.read_csv('./all_phys_data_points/phys_1.csv',index_col = 0)
    emg = pd.read_csv('./all_emg_data_points/emg_1.csv',index_col = 0)


    df = pd.DataFrame(data = extract_feature(point,emg,True,True),index = [0])

    for i in range(2,51):
        point = pd.read_csv('./all_phys_data_points/phys_'+str(i)+'.csv',index_col = 0)
        emg = pd.read_csv('./all_emg_data_points/emg_'+str(i)+'.csv',index_col = 0)
        df = df.append(extract_feature(point,emg,True,True),ignore_index = True)


    for col in df.columns:
        if col != 'label':
            df[col] = (df[col] - df[col].mean())/df[col].std()

    df.to_csv('./all_data_phys_emg.csv')

    df = pd.DataFrame(data = extract_feature(point,emg,True,False),index = [0])

    for i in range(2,51):
        point = pd.read_csv('./all_phys_data_points/phys_'+str(i)+'.csv',index_col = 0)
        emg = pd.read_csv('./all_emg_data_points/emg_'+str(i)+'.csv',index_col = 0)
        df = df.append(extract_feature(point,emg,True,False),ignore_index = True)


    for col in df.columns:
        if col != 'label':
            df[col] = (df[col] - df[col].mean())/df[col].std()

    df.to_csv('./all_data_phys.csv')
    
    df = pd.DataFrame(data = extract_feature(point,emg,False,True),index = [0])

    for i in range(2,51):
        point = pd.read_csv('./all_phys_data_points/phys_'+str(i)+'.csv',index_col = 0)
        emg = pd.read_csv('./all_emg_data_points/emg_'+str(i)+'.csv',index_col = 0)
        df = df.append(extract_feature(point,emg,False,True),ignore_index = True)


    for col in df.columns:
        if col != 'label':
            df[col] = (df[col] - df[col].mean())/df[col].std()

    df.to_csv('./all_data_emg.csv')
