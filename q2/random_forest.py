import sklearn as sk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('./train/emg_phys.csv',index_col =0)
val = pd.read_csv('./val/emg_phys.csv',index_col =0)

X_train = train.drop('label',axis = 1)
y_train = train['label']

X_test = val.drop('label',axis = 1)
y_test = val['label']

model = RandomForestClassifier()
cv_score = cross_val_score(model,X_train,y_train,cv=5).mean()

model.fit(X_train,y_train)
train_acc = model.score(X_train,y_train)
test_acc = model.score(X_test,y_test)

print()
print("EMG+PHYS:")
print()
print("CV Accuracy: " + str(cv_score))
print("Train Accuracy: " + str(train_acc))
print("Test Accuracy: " + str(test_acc))

train = pd.read_csv('./train/phys.csv',index_col =0)
val = pd.read_csv('./val/phys.csv',index_col =0)

X_train = train.drop('label',axis = 1)
y_train = train['label']

X_test = val.drop('label',axis = 1)
y_test = val['label']

model = RandomForestClassifier()
cv_score = cross_val_score(model,X_train,y_train,cv=5).mean()

model.fit(X_train,y_train)
train_acc = model.score(X_train,y_train)
test_acc = model.score(X_test,y_test)

print()
print("PHYS:")
print()
print("CV Accuracy: " + str(cv_score))
print("Train Accuracy: " + str(train_acc))
print("Test Accuracy: " + str(test_acc))

train = pd.read_csv('./train/emg.csv',index_col =0)
val = pd.read_csv('./val/emg.csv',index_col =0)

X_train = train.drop('label',axis = 1)
y_train = train['label']

X_test = val.drop('label',axis = 1)
y_test = val['label']

model = RandomForestClassifier()
cv_score = cross_val_score(model,X_train,y_train,cv=5).mean()

model.fit(X_train,y_train)
train_acc = model.score(X_train,y_train)
test_acc = model.score(X_test,y_test)

print()
print("EMG:")
print()
print("CV Accuracy: " + str(cv_score))
print("Train Accuracy: " + str(train_acc))
print("Test Accuracy: " + str(test_acc))
