#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score

data = pd.read_csv("C:/Users/rites/Desktop/SML/train.csv")

category=data.iloc[:,-1]
data=data.drop(data.columns[-1],axis=1)

lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
scores = lof.fit_predict(data)

non_outlier_indices = []
for score in scores:
    if score == 1:
        non_outlier_indices.append(True)
    else:
        non_outlier_indices.append(False)
non_outlier_indices = np.array(non_outlier_indices)

data = data[non_outlier_indices]
category = category[non_outlier_indices]

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
labelings=[]
for label in kmeans.labels_:
    labelings.append(label)
labelings = np.array(labelings)

data['clusters'] = labelings

x=data
y=category

model = LogisticRegression(max_iter=100000)
model.fit(x, y)
final_p = model.predict(x)

accuracy = accuracy_score(y, final_p)
print("Accuracy:", accuracy)

scores = cross_val_score(model, x, y, cv=5)
print("Cross-validation scores: {}".format(scores))
print("Average accuracy: {}".format(scores.mean()))


# In[3]:


data = pd.read_csv("C:/Users/rites/Downloads/test.csv")

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

labelings=[]
for label in kmeans.labels_:
    labelings.append(label)
labelings = np.array(labelings)

data['cluster'] = labelings

pred_final = model.predict(data)

data = pd.read_csv("C:/Users/rites/Downloads/test.csv")
x,y =data.shape
data.drop(data.iloc[:, 1:y], inplace=True, axis=1)

data['category']=pred_final

data.to_csv("C:/Users/rites/Desktop/SML/file.csv",index=False)


# In[ ]:




