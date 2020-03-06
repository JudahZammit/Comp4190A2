import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import sklearn as sk
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K

train = pd.read_csv('./train/emg_phys.csv',index_col =0)
val = pd.read_csv('./val/emg_phys.csv',index_col =0)

X_train = train.drop('label',axis = 1).to_numpy()
y_train = train['label'].to_numpy()
y_train = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_train.reshape(-1,1))

X_test = val.drop('label',axis = 1).to_numpy()
y_test = val['label'].to_numpy()
y_test = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_test.reshape(-1,1))


def evaluate_model(X_train,y_train,X_test,y_test):
    kf = KFold(n_splits=5,shuffle=True)

    DR_RATE = 0.5
    EPOCHS = 20
    inputs = keras.layers.Input(X_train.shape[1])

    x = keras.layers.Dense(500)(inputs)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(DR_RATE)(x)

    x = keras.layers.Dense(500)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(DR_RATE)(x)

    out = keras.layers.Dense(5,activation = 'softmax')(x)
    model = keras.models.Model(inputs,out)

    cv_acc = 0
    for train_index,test_index in kf.split(X_train):
        X_train_split = np.take(X_train,train_index,axis = 0)
        X_test_split = np.take(X_train,test_index,axis = 0)
    
        y_train_split = np.take(y_train,train_index,axis =0)
        y_test_split = np.take(y_train,test_index,axis = 0)
        
        K.clear_session()
        model.compile('Adam',loss = 'categorical_crossentropy',metrics = ['acc'])
        model.fit(X_train_split,y_train_split,
              epochs = EPOCHS,batch_size = X_train_split.shape[0])
        cv_acc = cv_acc + model.evaluate(X_test_split,y_test_split)[1]
    cv_acc = cv_acc/5

    K.clear_session()
    model.compile('Adam',loss = 'categorical_crossentropy',metrics = ['acc'])
    model.fit(X_train,y_train,
            epochs = EPOCHS,batch_size = X_train.shape[0])
    train_acc = model.evaluate(X_train,y_train)[1]
    test_acc = model.evaluate(X_test,y_test)[1]

    return cv_acc,train_acc,test_acc

train = pd.read_csv('./train/phys.csv',index_col =0)[['orin_y_mad','orin_y_median','label']]
val = pd.read_csv('./val/phys.csv',index_col =0)[['orin_y_mad','orin_y_median','label']]

X_train = train.drop('label',axis = 1).to_numpy()
y_train = train['label'].to_numpy()
y_train = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_train.reshape(-1,1))

X_test = val.drop('label',axis = 1).to_numpy()
y_test = val['label'].to_numpy()
y_test = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_test.reshape(-1,1))
single_avg_acc,single_train_acc,single_test_acc = evaluate_model(X_train,y_train,X_test,y_test)

train = pd.read_csv('./train/emg.csv',index_col =0)
val = pd.read_csv('./val/emg.csv',index_col =0)

X_train = train.drop('label',axis = 1).to_numpy()
y_train = train['label'].to_numpy()
y_train = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_train.reshape(-1,1))

X_test = val.drop('label',axis = 1).to_numpy()
y_test = val['label'].to_numpy()
y_test = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_test.reshape(-1,1))
e_avg_acc,e_train_acc,e_test_acc = evaluate_model(X_train,y_train,X_test,y_test)

train = pd.read_csv('./train/phys.csv',index_col =0)
val = pd.read_csv('./val/phys.csv',index_col =0)

X_train = train.drop('label',axis = 1).to_numpy()
y_train = train['label'].to_numpy()
y_train = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_train.reshape(-1,1))

X_test = val.drop('label',axis = 1).to_numpy()
y_test = val['label'].to_numpy()
y_test = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_test.reshape(-1,1))
p_avg_acc,p_train_acc,p_test_acc = evaluate_model(X_train,y_train,X_test,y_test)

train = pd.read_csv('./train/emg_phys.csv',index_col =0)
val = pd.read_csv('./val/emg_phys.csv',index_col =0)

X_train = train.drop('label',axis = 1).to_numpy()
y_train = train['label'].to_numpy()
y_train = sk.preprocessing.OneHotEncoder(sparse = False
        ).fit_transform(y_train.reshape(-1,1))

X_test = val.drop('label',axis = 1).to_numpy()
y_test = val['label'].to_numpy()
y_test = sk.preprocessing.OneHotEncoder(sparse = False
    ).fit_transform(y_test.reshape(-1,1))
pe_avg_acc,pe_train_acc,pe_test_acc = evaluate_model(X_train,y_train,X_test,y_test)

print()
print("PHYS+EMG:")
print()
print("CV Accuracy: " + str(pe_avg_acc))
print("Train Accuracy: " + str(pe_train_acc))
print("Test Accuracy: " + str(pe_test_acc))

print()
print("Y-Orintation:")
print()
print("CV Accuracy: " + str(single_avg_acc))
print("Train Accuracy: " + str(single_train_acc))
print("Test Accuracy: " + str(single_test_acc))

print()
print("EMG:")
print()
print("CV Accuracy: " + str(e_avg_acc))
print("Train Accuracy: " + str(e_train_acc))
print("Test Accuracy: " + str(e_test_acc))

print()
print("PHYS:")
print()
print("CV Accuracy: " + str(p_avg_acc))
print("Train Accuracy: " + str(p_train_acc))
print("Test Accuracy: " + str(p_test_acc))
