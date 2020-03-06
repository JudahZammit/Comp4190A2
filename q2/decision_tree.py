import sklearn as sk
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import graphviz

train = pd.read_csv('./train/emg_phys.csv',index_col =0)
val = pd.read_csv('./val/emg_phys.csv',index_col =0)

X_train = train.drop('label',axis = 1)
y_train = train['label']

X_test = val.drop('label',axis = 1)
y_test = val['label']

model = DecisionTreeClassifier()
cv_score = cross_val_score(model,X_train,y_train,cv=5).mean()

model.fit(X_train,y_train)
train_acc = model.score(X_train,y_train)
test_acc = model.score(X_test,y_test)

dot_data = sk.tree.export_graphviz(model, out_file=None, 
                      feature_names=X_train.columns,    
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render('tree') 

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

model = DecisionTreeClassifier()
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

model = DecisionTreeClassifier()
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

train = pd.read_csv('./train/phys.csv',index_col =0)[['orin_y_mad','orin_y_median','label']]
val = pd.read_csv('./val/phys.csv',index_col =0)[['orin_y_mad','orin_y_median','label']]

X_train = train.drop('label',axis = 1)
y_train = train['label']

X_test = val.drop('label',axis = 1)
y_test = val['label']

model = DecisionTreeClassifier()
cv_score = cross_val_score(model,X_train,y_train,cv=5).mean()

model.fit(X_train,y_train)
train_acc = model.score(X_train,y_train)
test_acc = model.score(X_test,y_test)

print()
print("Y-Orintation:")
print()
print("CV Accuracy: " + str(cv_score))
print("Train Accuracy: " + str(train_acc))
print("Test Accuracy: " + str(test_acc))
