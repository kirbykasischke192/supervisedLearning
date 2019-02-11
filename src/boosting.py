#https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
#https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models
#https://www.datacamp.com/community/tutorials/adaboost-classifier-python

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn import model_selection, metrics
import time as time

satdata = pd.read_csv("./../data/sat_data.csv")
pendata = pd.read_csv("./../data/pen_data.csv")

# sat_num_rows = satdata.index._stop
# pen_num_rows = pendata.index._stop

# pen_x = []
# sat_x = []
# pen_test_error = []
# sat_test_error = []



# sat_num_items = int(sat_num_rows*i*.05)
# pen_num_items = int(pen_num_rows*i*.05)
# pen_x.append(pen_num_items)
# sat_x.append(sat_num_items)

Xsat = satdata.iloc[:, :-1].values
ysat = satdata.iloc[:, 36].values

Xpen = pendata.iloc[:, :-1].values
ypen = pendata.iloc[:, 16].values

from sklearn.model_selection import train_test_split  
X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split(Xsat, ysat, test_size=0.20)
X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(Xpen, ypen, test_size=0.20)

from sklearn.preprocessing import StandardScaler  
pen_scaler = StandardScaler()
sat_scaler = StandardScaler()
pen_scaler.fit(X_train_pen)
sat_scaler.fit(X_train_sat)

X_train_pen = pen_scaler.transform(X_train_pen)
X_train_sat = sat_scaler.transform(X_train_sat)
X_test_pen = pen_scaler.transform(X_test_pen) 
X_test_sat = sat_scaler.transform(X_test_sat)

pen_train_error = []
pen_test_error = []
pen_test_accuracy = []
sat_train_error = []
sat_test_error = []
sat_test_accuracy = []
pen_train_size = []
sat_train_size = []

datasplits = np.linspace(.05, 1.0, 20, endpoint=True)     
seed = 7
num_trees = 30
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
pen_model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
#pen_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
penT1 = time.time()   
pen_model.fit(X_train_pen, y_train_pen)
penT2 = time.time()
#pen_train_size, pen_train_score, pen_test_score = model_selection.learning_curve(
    #pen_model, X_train_pen, y_train_pen, train_sizes=datasplits, shuffle=True, cv=3,
    #scoring='accuracy')
#y_pred_pen_train = pen_model.predict(X_train_pen)
#y_pred_pen_test = pen_model.predict(X_test_pen)
#pen_results = model_selection.cross_val_score(pen_model, X_train_pen, y_train_pen, cv=kfold)

sat_model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
#sat_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
satT1 = time.time()
sat_model.fit(X_train_sat, y_train_sat)
satT2 = time.time()
#sat_train_size, sat_train_score, sat_test_score = model_selection.learning_curve(
    #sat_model, X_train_sat, y_train_sat, train_sizes=datasplits, shuffle=True, cv=3, 
    #scoring='accuracy')
#y_pred_sat_train = sat_model.predict(X_train_sat)
#y_pred_sat_test = sat_model.predict(X_test_sat)
#sat_results = model_selection.cross_val_score(sat_model, X_train_sat, y_train_sat, cv=kfold)
#pen_test_accuracy.append(metrics.accuracy_score(y_test_pen, y_pred_pen_test))
#sat_test_accuracy.append(metrics.accuracy_score(y_test_sat, y_pred_sat_test))

# pen_train_error.append(metrics.mean_absolute_error(y_train_pen, y_pred_pen_train))
# pen_test_error.append(metrics.mean_absolute_error(y_test_pen, y_pred_pen_test))
# sat_train_error.append(metrics.mean_absolute_error(y_train_sat, y_pred_sat_train))
# sat_test_error.append(metrics.mean_absolute_error(y_test_sat, y_pred_sat_test))

print(satT2-satT1)
print(penT2-penT1)

# pen_train_accuracy = np.mean(pen_train_score, axis=1)
# pen_test_accuracy = np.mean(pen_test_score, axis=1)
# sat_train_accuracy = np.mean(sat_train_score, axis=1)
# sat_test_accuracy = np.mean(sat_test_score, axis=1)

# pen_train_error = [1 - accuracy for accuracy in pen_train_accuracy]
# pen_test_error = [1 - accuracy for accuracy in pen_test_accuracy]
# sat_train_error = [1 - accuracy for accuracy in sat_train_accuracy]
# sat_test_error = [1 - accuracy for accuracy in sat_test_accuracy]

# plt.figure(figsize=(12, 6))  
# plt.plot(pen_train_size, pen_train_error, label='Training Error', color='red', linestyle='solid', marker='')
# plt.plot(pen_train_size, pen_test_error, label='Testing Error', color='green', linestyle='dashed', marker='')
# plt.title('Error Rate vs Training Set Size')  
# plt.xlabel('Training Set Size')  
# plt.ylabel('Error')
# plt.legend()
# plt.show()
# plt.plot(sat_train_size, sat_train_error, label='Training Error', color='red', linestyle='solid', marker='')
# plt.plot(sat_train_size, sat_test_error, label='Testing Error', color='green', linestyle='dashed', marker='')
# plt.title('Error Rate vs Training Set Size')  
# plt.xlabel('Training Set Size')  
# plt.ylabel('Error')
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))  
# plt.plot(treenums, pen_test_error, label='Testing Error', color='red', linestyle='solid', marker='')
# plt.title('Error Rate vs Number of Estimators')  
# plt.xlabel('Number of Estimators')  
# plt.ylabel('Error')
# plt.show()
# plt.figure(figsize=( 12, 6))
# plt.plot(treenums, sat_test_error, label='Testing Error', color='red', linestyle='solid', marker='')
# plt.title('Error Rate vs Number of Estimators')  
# plt.xlabel('Number of Estimators')  
# plt.ylabel('Error')
# plt.show()

#print(sat_results.mean())
#print(pen_results.mean())
#print(*sat_sqerror)
#print(*pen_sqerror)