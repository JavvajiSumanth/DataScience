# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:32:12 2019

@author: Sankar
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m

#import the dataset
dataset=pd.read_csv('train.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,20:21].values

'''#categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
Y=onehotencoder.fit_transform(Y).toarray()'''



#splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/4,random_state=0)


#Diimensionality Reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=4)
X_train=lda.fit_transform(X_train,Y_train)
X_test=lda.transform(X_test)

from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

s=int(m.sqrt(cm.size))
sum1=0
sum2=0 

for i in range(0,s):
    for j in range(0,s):
            if i==j:
                sum1 = sum1 + cm[i][j]
            else:
                sum2 = sum2 + cm[i][j]
                
total=sum1+sum2                
Accuracy=(sum1/total)*100            
print("The accuracy for the given test set is " + str(float(Accuracy)) + "%")


dataset1=pd.read_csv('')