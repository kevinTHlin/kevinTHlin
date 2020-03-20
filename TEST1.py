#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import datasets
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'

iris = datasets.load_iris()
features = iris.data
target = iris.target

model = DecisionTreeClassifier()
model.fit(features, target)
dot_data = tree.export_graphviz(model, 
                  feature_names=iris.feature_names,  
                  class_names=iris.target_names,  
                  filled=True, rounded=True,  
                  special_characters=True,
                   out_file=None,
                           )
graph = graphviz.Source(dot_data)
graph


# 
