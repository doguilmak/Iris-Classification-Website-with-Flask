# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 18:57:02 2021

@author: doguilmak

Review dependent and independent variables

"""
#%%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

#%%

start = time.time()
df = pd.read_excel('Iris.xls')

IRIS=df['iris']

print("-------\n",df.isnull().sum())
print("-------\n", list(df.columns))
print("-------\n", df.head(10))
print("-------\n", df.describe().T)
print("-------\n", df.info())
print('-------\n', IRIS.value_counts(), '\n')

plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')    
sns.histplot(data=IRIS)
plt.title("Iris Classes on Histogram")
plt.xlabel("Iris Classes")
plt.ylabel("Plants in Total")
#plt.savefig('hist_iris')
plt.show()

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
