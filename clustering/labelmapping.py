#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install munkres
import pickle
# https://gist.github.com/siolag161/dc6e42b64e1bde1f263b


# In[2]:


'''
If we use (external) classification evalutation measures like F1 or 
accuracy for clustering evaluation, problems may arise. 
One way to fix is to perform label matching.
Here we performs kmeans clustering on the Iris dataset and proceed to use 
the Hungarian (Munkres) algorithm to correct the mismatched labeling. 
'''

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix,precision_score,recall_score

from munkres import Munkres


# In[3]:


def make_cost_matrix(c1, c2):
    """
    """
    uc1 = np.unique(c1)
    uc2 = np.unique(c2)
    l1 = uc1.size
    l2 = uc2.size
    assert(l1 == l2 and np.all(uc1 == uc2))

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size
    return m

def translate_clustering(clt, mapper):
    return np.array([ mapper[i] for i in clt ])

def accuracy(cm):
    """computes accuracy from confusion matrix"""
    return np.trace(cm, dtype=float) / np.sum(cm)

# def label_matching(classes):
    
#     # classes are true labels,
    
#     """entry point"""
#     dataset = datasets.load_iris() # loads the iris dataset
#     data, classes = dataset.data, dataset.target # data and true labels
#     algo = KMeans(n_clusters=3, random_state = 0)
    
#     labels = algo.fit(data).labels_ # performs the algo and get the predicted labels
#     num_labels = len(np.unique(classes))
#     print("classes",classes)

#     cm = confusion_matrix(classes, labels, labels=range(num_labels)) # gets the confusion matrix
#     print("---------------------\nold confusion matrix:\n" " %s\naccuracy: %.2f" % (str(cm), accuracy(cm)))

#     cost_matrix = make_cost_matrix(labels, classes)
#     print(labels)
#     print(len(labels))
    
#     m = Munkres()
#     indexes = m.compute(cost_matrix)
#     mapper = { old: new for (old, new) in indexes }

#     print("---------------------\nmapping:")
#     for old, new in mapper.items():
#         print("map: %s --> %s" %(old, new))

#     new_labels = translate_clustering(labels, mapper)
#     new_cm = confusion_matrix(classes, new_labels, labels=range(num_labels))
#     print("---------------------\nnew confusion matrix:\n" \
#           " %s\naccuracy: %.2f" % (str(new_cm), accuracy(new_cm)))


# if __name__ == "__main__":
#     label_matching()


# In[8]:


def label_matching(classes,labels):   # Gtruth,pred
    
    # classes are true labels,
    
    """entry point"""
    #dataset = datasets.load_iris() # loads the iris dataset
    #data, classes = dataset.data, dataset.target # data and true labels
    #algo = KMeans(n_clusters=3, random_state = 0)
    
    #labels = algo.fit(data).labels_ # performs the algo and get the predicted labels
    num_labels = len(np.unique(classes))
#    print("classes",classes)

    cm = confusion_matrix(classes, labels, labels=range(num_labels)) # gets the confusion matrix
#    print("---------------------\nold confusion matrix:\n" " %s\naccuracy: %.2f" % (str(cm), accuracy(cm)))

    cost_matrix = make_cost_matrix(labels, classes)
    pr = precision_score(classes, labels,average='macro')
    re = recall_score(classes, labels,average='macro')

    m = Munkres()
    indexes = m.compute(cost_matrix)
    mapper = {old: new for (old, new) in indexes }

    new_labels = translate_clustering(labels, mapper)
    new_cm = confusion_matrix(classes, new_labels, labels=range(num_labels))

    new_pr = precision_score(classes, new_labels,average='macro')
    new_re = recall_score(classes, new_labels,average='macro')
    
    return new_labels,cm,accuracy(cm),pr,re,new_cm,accuracy(new_cm),new_pr,new_re # Returns new prediction labels

if __name__ == "__main__":
#     predictions = [1,0,0,1]
#     groundtruth = [0,1,0,1]     

    # Mia Sara [['date of birth'], ['cast member_1', 'occupation_2', 'work period (start)'], ['educated at'], ['cast member_2', 'occupation_1', 'cast member', 'work period (start)']] 4
    # {'work period (start)', 'educated at', 'cast member_1','cast member_2', 'occupation_1', 'date of birth'}
    # Clusters [2 3 1 0 1]
#     predictions = [2,3,1,0,1
#     groundtruth = [0,1,0,0,2]
                   
                   
#     predictions = [1,3,2,0,0,2,0,1,0]
#     groundtruth = [0, 1, 1, 1, 2, 3, 3, 3, 3]
    
    predictions = [3,2,2,1,1,1,1,3,3,1,2,5,0,4]
    groundtruth = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 5]
 
                   
    
    # Rituparno Ghosh [['date of birth', 'date of death', 'occupation', 'occupation'], ['educated at'], ['date of death', 'residence', 'place of death', 'place of birth'], ['award received'], ['date of death', 'place of death'], ['place of death']] 6
# {'place of death', 'place of birth', 'date of death', 'educated at', 'residence', 'occupation', 'award received', 'date of birth'}
# Clusters [3 5 3 2 4 1 0 2]
    
    print(label_matching(groundtruth,predictions))
