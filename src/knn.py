#https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.neighbors as neighbors
import time as time

satdata = pd.read_csv("./../data/sat_data.csv")
pendata = pd.read_csv("./../data/pen_data.csv")

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

from sklearn.neighbors import KNeighborsClassifier
sat_accuracy = []
pen_accuracy = []

leafsize = range(1, 61)
algorithms = ['ball_tree', 'kd_tree', 'brute']

sat_classifier = KNeighborsClassifier(n_neighbors=6)
pen_classifier = KNeighborsClassifier(n_neighbors=1)
satT1 = time.time()   
sat_classifier.fit(X_train_sat, y_train_sat)
satT2 = time.time()
penT1 = time.time()
pen_classifier.fit(X_train_pen, y_train_pen)
penT2 = time.time()

print(satT2-satT1)
print(penT2-penT1)

y_pred_sat = sat_classifier.predict(X_test_sat)
y_pred_pen = pen_classifier.predict(X_test_pen)
sat_accuracy.append(metrics.accuracy_score(y_test_sat, y_pred_sat))
pen_accuracy.append(metrics.accuracy_score(y_test_pen, y_pred_pen))

pen_error = [1 - accuracy for accuracy in pen_accuracy]
sat_error = [1 - accuracy for accuracy in sat_accuracy]
    
# plt.figure(figsize=(12, 6))  
# plt.plot(leafsize, pen_error, color='red', linestyle='solid', marker='')
# for i in range(3):
#     plt.bar(algorithms[i], pen_error[i])
# plt.title('Error Rate vs Leaf Size')  
# plt.xlabel('Leaf Size')  
# plt.ylabel('Error')
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.plot(leafsize, sat_error, color='red', linestyle='solid', marker='')
# for i in range(3):
#     plt.bar(algorithms[i], sat_error[i])
# plt.title('Error Rate vs Leaf Size')  
# plt.xlabel('Leaf Size')  
# plt.ylabel('Error')
# plt.show()

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test_sat, y_pred_sat))
# print(confusion_matrix(y_test_pen, y_pred_pen))
# print(classification_report(y_test_sat, y_pred_sat))
# print(classification_report(y_test_pen, y_pred_pen))