#https://stackabuse.com/decision-trees-in-python-with-scikit-learn/

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import metrics  
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

pen_train_error = []
pen_test_error = []
sat_train_error = []
sat_test_error = []
sat_accuracy = []
pen_accuracy = []

from sklearn.tree import DecisionTreeClassifier  
max_depth_list = np.linspace(2, 50, 49, endpoint=True)

sat_classifier = DecisionTreeClassifier(max_depth=10, min_samples_split=10)
pen_classifier = DecisionTreeClassifier(max_depth=15, min_samples_split=2) 
satT1 = time.time()   
sat_classifier.fit(X_train_sat, y_train_sat)
satT2 = time.time()
penT1 = time.time()
pen_classifier.fit(X_train_pen, y_train_pen)
penT2 = time.time()

# print(satT2-satT1)
# print(penT2-penT1)

y_pred_sat_test = sat_classifier.predict(X_test_sat)
y_pred_pen_test = pen_classifier.predict(X_test_pen)

pen_accuracy.append(metrics.accuracy_score(y_test_pen, y_pred_pen_test))
sat_accuracy.append(metrics.accuracy_score(y_test_sat, y_pred_sat_test))

pen_test_error = [1 - accuracy for accuracy in pen_accuracy]
sat_test_error = [1 - accuracy for accuracy in sat_accuracy]

print(pen_test_error)
print(sat_test_error)
    
# plt.figure(figsize=(12, 6))  
# plt.plot(max_depth_list, pen_test_error, label='Testing Error', color='red', linestyle='solid', marker='')
# plt.title('Error Rate vs Minimum Samples to Split Internal Node')  
# plt.xlabel('Minimum Samples')  
# plt.ylabel('Error')
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.plot(max_depth_list, sat_test_error, label='Testing Error', color='red', linestyle='solid', marker='')
# plt.title('Error Rate vs Minimum Samples to Split Internal Node')  
# plt.xlabel('Minimum Samples')  
# plt.ylabel('Error')
# plt.show()

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test_sat, y_pred_sat))
# print(confusion_matrix(y_test_pen, y_pred_pen))
# print(classification_report(y_test_sat, y_pred_sat))
# print(classification_report(y_test_pen, y_pred_pen))

# from sklearn import metrics  
# print('Sat Mean Absolute Error:', metrics.mean_absolute_error(y_test_sat, y_pred_sat))  
# print('Sat Mean Squared Error:', metrics.mean_squared_error(y_test_sat, y_pred_sat))  
# print('Sat Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_sat, y_pred_sat)))  
# print('Pen Mean Absolute Error:', metrics.mean_absolute_error(y_test_pen, y_pred_pen))  
# print('Pen Mean Squared Error:', metrics.mean_squared_error(y_test_pen, y_pred_pen))  
# print('Pen Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_pen, y_pred_pen)))  
