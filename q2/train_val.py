import pandas as pd


emg_phys = pd.read_csv('./all_data_phys_emg.csv',index_col =0)
emg = pd.read_csv('./all_data_emg.csv',index_col =0)
phys = pd.read_csv('./all_data_phys.csv',index_col = 0)

val_rows = [4,14,24,34,44]

train_emg_phys = emg_phys.drop([emg_phys.index[3],emg_phys.index[14],
        emg_phys.index[25],emg_phys.index[36],emg_phys.index[47]])
val_emg_phys = emg_phys.iloc[val_rows]

train_emg = emg.drop([emg.index[3],emg.index[14],
    emg.index[25],emg.index[36],emg.index[47]])
val_emg = emg.iloc[val_rows]

train_phys = phys.drop([phys.index[3],phys.index[14],
    phys.index[25],phys.index[36],phys.index[47]])
val_phys = phys.iloc[val_rows]

train_emg_phys.to_csv('./train/emg_phys.csv')
val_emg_phys.to_csv('./val/emg_phys.csv')

train_emg.to_csv('./train/emg.csv')
val_emg.to_csv('./val/emg.csv')

train_phys.to_csv('./train/phys.csv')
val_phys.to_csv('./val/phys.csv')
