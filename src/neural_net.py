#https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
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
  
max_depth_list = np.linspace(2, 50, 49, endpoint=True)
layer_size = [10,20,30,40,50,60,70,80,90,100]
hidden_layers = [(10,10,10), (20,20,20), (30,30,30), (40,40,40), (50,50,50),
    (60,60,60), (70,70,70), (80,80,80),
    (90,90,90), (100,100,100)]
functions = ['identity', 'logistic', 'tanh', 'relu']
for function in functions:
    sat_classifier = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, activation=function)
    pen_classifier = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, activation=function)
    #satT1 = time.time()   
    sat_classifier.fit(X_train_sat, y_train_sat.ravel())
    #satT2 = time.time()
    #penT1 = time.time()
    pen_classifier.fit(X_train_pen, y_train_pen.ravel())
    #penT2 = time.time()
    y_pred_sat_test = sat_classifier.predict(X_test_sat)
    y_pred_pen_test = pen_classifier.predict(X_test_pen)

    pen_accuracy.append(metrics.accuracy_score(y_test_pen, y_pred_pen_test))
    sat_accuracy.append(metrics.accuracy_score(y_test_sat, y_pred_sat_test))

# print(satT2-satT1)
# print(penT2-penT1)

pen_test_error = [1 - accuracy for accuracy in pen_accuracy]
sat_test_error = [1 - accuracy for accuracy in sat_accuracy]
    
plt.figure(figsize=(12, 6))  
#plt.plot(layer_size, pen_test_error, label='Testing Error', color='red', linestyle='solid', marker='')
for i in range(4):
        plt.bar(functions[i], pen_test_error[i])
plt.title('Error Rate vs Activation Function')  
plt.xlabel('Activation Function')  
plt.ylabel('Error')
plt.show()
plt.figure(figsize=(12, 6))
#plt.plot(layer_size, sat_test_error, label='Testing Error', color='red', linestyle='solid', marker='')
for i in range(4):
    plt.bar(functions[i], sat_test_error[i])
plt.title('Error Rate vs Activation Function')  
plt.xlabel('Activation Function')  
plt.ylabel('Error')
plt.show()

# from sklearn.neural_network import MLPClassifier  
# sat_mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
# sat_mlp.fit(X_train_sat, y_train_sat.ravel())
# pen_mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
# pen_mlp.fit(X_train_pen, y_train_pen.ravel())

# y_pred_sat = sat_mlp.predict(X_test_sat)
# y_pred_pen = pen_mlp.predict(X_test_pen)

# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test_sat, y_pred_sat))
# print(confusion_matrix(y_test_pen, y_pred_pen))
# print(classification_report(y_test_sat, y_pred_sat))
# print(classification_report(y_test_pen, y_pred_pen))
# print(accuracy_score(y_test_pen, y_pred_pen))
# print(accuracy_score(y_test_sat, y_pred_sat))

