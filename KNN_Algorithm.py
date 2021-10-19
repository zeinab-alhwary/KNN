
# KNN algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris=load_iris()

print(type(iris))
print(iris.feature_names)      
#print(iris.target)
print(np.unique(iris.target))   #classification types of flower


from sklearn.model_selection import train_test_split  #split model to train & test

x=iris.data         # 4 feature
y=iris.target       # classification

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20)

#print(x_train.shape)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

print(knn.fit(x_train,y_train))


y_pred=knn.predict(x_test)
print(len(y_pred))

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

#print(knn.predict([[1,3,2,5]]))
