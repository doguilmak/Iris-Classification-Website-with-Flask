# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:31:37 2021

@author: doguilmak

Deploying classification algorithm with pickle.

"""
#%%

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

#%%

df = pd.read_excel('Iris.xls')

label_encoder = LabelEncoder()

categorical_features = [feature for feature in df.columns 
                        if df[feature].dtypes == 'O']

for feature in categorical_features:
    print(feature, list(df[feature].unique()), 
          list(label_encoder.fit_transform(df[feature].unique())), "\n")

categorical_features = [feature for feature in df.columns 
                        if df[feature].dtypes == 'O']

for feature in categorical_features:
    df[feature]=label_encoder.fit_transform(df[feature])

class_map = {0: "Iris-setosa", 
            1: "Iris-versicolor", 
            2: "Iris-virginica"}

def predict_class(sepal_length, sepal_width, petal_length, petal_width):

    # Read the machine learning model
    pickle_file = open('classifier.pkl', 'rb')     
    classifier = pickle.load(pickle_file)

    y_predict = classifier.predict([[sepal_length, 
                                     sepal_width, 
                                     petal_length, 
                                     petal_width]])[0]

    return class_map[y_predict]

print(predict_class(5.1, 3.5, 1.4, 0.2))
