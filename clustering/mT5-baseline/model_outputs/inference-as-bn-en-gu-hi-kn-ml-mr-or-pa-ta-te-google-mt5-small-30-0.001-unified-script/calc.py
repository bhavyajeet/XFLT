import json
import pandas as pd
import re

import pickle
# https://gist.github.com/siolag161/dc6e42b64e1bde1f263b

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from munkres import Munkres


def make_cost_matrix(c1, c2):
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
    return np.trace(cm, dtype=float) / np.sum(cm)


def label_matching(classes,labels):   # Gtruth,pred
    
    """entry point"""
    num_labels = len(np.unique(classes))

    cm = confusion_matrix(classes, labels, labels=range(num_labels)) # gets the confusion matrix

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

"""
if __name__ == "__main__":
    
    predictions = [3,2,2,1,1,1,1,3,3,1,2,5,0,4]
    groundtruth = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 5]
    print(label_matching(groundtruth,predictions))

"""



srcfile = './test-src.txt'
reffile = './test-ref.txt'
predfile = 'test-predicted-epoch-9.txt'


predlines = open(predfile).readlines()
reffile = open(reffile).readlines()


def get_all (flist):
    rlist = []
    for i in flist:
        il = i.split('<T>')
        il = il[:-1]
        new = [x.split('<R>')[1].strip() for x in il]
        rlist += new 
    return rlist


def getreflist(flist):
    c = []
    curr = 0 
    for i in flist:
        il = i.split('<T>')
        il = il[:-1]
        new = [x.split('<R>')[1].strip() for x in il]
        newlist = [curr] * len(new)
        c += newlist
        curr += 1

    return c


def getpredlist(flist,klist,cllist):
    assert len(klist) == len(cllist)

    clean_list = []

    for i in flist:
        il = i.split('<T>')
        il = il[:-1]
        new = [x.split('<R>')[1].strip() for x in il]
        clean_list.append(new)

    #print ("flist: ",clean_list)
    #print ("klist: ",klist)
    #print ("cls: ",cllist)

    c = [-1] * len(cllist)
    
    curr = 0
    for i in flist:
        il = i.split('<T>')
        il = il[:-1]
        new = [x.split('<R>')[1].strip() for x in il]
        for rl in new:
            if rl in klist:
                vr = klist.index(rl)
                klist[vr] = 'random relation'
                c[vr] = curr
            else :
                vr = c.index(-1)
                klist[vr] = 'random relation'
                c[vr] = curr
        curr += 1

    #print (c)
    return c



macro_pr = []
macro_rec = []
macro_pr_old = []
macro_rec_old = []
facts_len = 0
cnt = 0
weighted_acc =[]
missed = []



count =0
for it in range(len(predlines)):
    c_pred = predlines[it]
    c_ref = reffile[it]

    if c_pred == c_ref:
        print ("yes")

    c_pred = c_pred.replace('<pad>','')
    c_ref = c_ref.replace('<pad>','')
    
    c_pred = re.sub(' +', ' ', c_pred)  
    c_ref = re.sub(' +', ' ', c_ref)  

    clus_pred = c_pred.split('</s>')[0].strip().split('<BR>')
    clus_ref = c_ref.split('</s>')[0].strip().split('<BR>')

    pred_lst = []
    ref_lst = []

    for i in clus_pred:
        if i != '':
            pred_lst.append(i.strip())

    for i in clus_ref:
        if i != '':
            ref_lst.append(i.strip())
    
    ll = get_all(ref_lst)
    pk = get_all(pred_lst)

    cls = getreflist(ref_lst)

    if len(ll) != len(pk) or len(np.unique(ll)) != len(np.unique(pk)):
        cnt += 1
        #pass 
    else :
        pred = getpredlist(pred_lst,ll,cls)
        uc1 = np.unique(pred)
        uc2 = np.unique(cls)
        l1 = uc1.size
        l2 = uc2.size
        if (l1 == l2 and np.all(uc1 == uc2)):
            cnt += 1
            new_clusters,cm,old_acc,old_pr,old_re,new_cm,new_acc,new_pr,new_re  = label_matching(cls,pred)
            macro_pr.append(new_pr)
            macro_rec.append(new_re)
            weighted_acc.append(new_acc * len(cls))
            facts_len+= len(cls)
            macro_pr_old.append(old_pr)
            macro_rec_old.append(old_re)

        else :
            #print (pred,cls)
            if l1 > l2 :
                tofill = [x for x in uc1 if x not in uc2]
                unq = []
                for w in range(len(cls)) :
                    el  = cls[w]
                    if el not in unq :
                        unq.append(el)
                    elif len(tofill):
                        cls[w] = tofill[0]
                        tofill.pop(0)
            elif l2 > l1:
                tofill = [x for x in uc2 if x not in uc1]
                unq = []
                for w in range(len(pred)) :
                    el  = pred[w]
                    if el not in unq :
                        unq.append(el)
                    elif len(tofill):
                        pred[w] = tofill[0]
                        tofill.pop(0)

            uc1 = np.unique(pred)
            uc2 = np.unique(cls)
            l1 = uc1.size
            l2 = uc2.size
            if (l1 == l2 and np.all(uc1 == uc2)):
                pass
            else :
                print (pred,cls)
                print (uc1,uc2) 

            new_clusters,cm,old_acc,old_pr,old_re,new_cm,new_acc,new_pr,new_re  = label_matching(cls,pred)
            macro_pr.append(new_pr)
            macro_rec.append(new_re)
            weighted_acc.append(new_acc * len(cls))
            facts_len+= len(cls)
            macro_pr_old.append(old_pr)
            macro_rec_old.append(old_re)
            cnt += 1
            count += 1

    """
    if len(ll) != len(pk):
        print (len(ll),len(pk))
        print(ll)
        print (pk)

    """

macro_pr_total = pd.Series(macro_pr).sum()/cnt
macro_rec_total = pd.Series(macro_rec).sum()/cnt
weighted_acc_total = pd.Series(weighted_acc).sum()/facts_len
macro_pr_old_total = pd.Series(macro_pr_old).sum()/cnt
macro_rec_old_total = pd.Series(macro_rec_old).sum()/cnt

scores = {}
scores['macro_pr'] = round(macro_pr_total*100,2)
scores['macro_rec'] = round(macro_rec_total*100,2)
scores['weighted_acc'] = round(weighted_acc_total*100,2)
scores['macro_pr_old'] = round(macro_pr_old_total*100,2)
scores['macro_rec_old'] = round(macro_rec_old_total*100,2)
print(scores)


print (count)
