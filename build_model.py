# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:12:48 2021

@author: doguilmak

Building K-NN classification algorithm for Iris data set.

"""
#%%

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

#%%

start = time.time()
df = pd.read_excel('Iris.xls')

label_encoder = LabelEncoder()

categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
for feature in categorical_features:
    df[feature]=label_encoder.fit_transform(df[feature])

X = df.drop("iris", axis=1)
y = df["iris"]

#%%
# K-NN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score

knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski')
knn.fit(X, y)

kfold = KFold(random_state=42, shuffle=True)
cv_results = cross_val_score(knn, X, y, cv=kfold, scoring="accuracy")

print(cv_results.mean())
print(cv_results.std())

import pickle

pickle_file = open('classifier.pkl', 'ab')
pickle.dump(knn, pickle_file)                     
pickle_file.close()

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
