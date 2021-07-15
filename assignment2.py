#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:37:00 2021

@author: sumanroy & Shiuli Subhra Ghosh 
"""
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def data_from_txt(s):    
    f = open(s, "r")
    a = f.readlines()
    b = a[0:3]
    a = a[3:]
    f.close()
    a = [i.replace('\n','') for i in a]
    b = [i.replace ('\n','') for i in b]
    n = int(b[1]) # No of words in dictionary
    messi = defaultdict(list)
    for i in a:
        x = i.split(" ")
        messi[float(x[0])].append(float(x[1]))
    ronaldo = defaultdict()
    for m in messi.keys():
        t = [0]*(n+1)
        for i in messi[m]:
            t[i] = 1
        ronaldo[m] = t[:]
    mbappe=pd.DataFrame.from_dict(ronaldo)#converting into a dataframe
    mbappe=mbappe.iloc[1:,:]
    return(mbappe)
    
def jaccard(mbappe):
    pele=mbappe.to_numpy() #converting into a numpy array
    cryuff=ss.csc_matrix(pele) #sparse column matrix to calculate jaccard distance
    jm = pairwise_distances(np.transpose(cryuff.toarray()), metric='jaccard')
    return(jm)
    
def kmeans(data,jm,k=5,max_iter=20):
    centroids={}
    cluster={}
    doc_list= list(range(data.shape[1])) #getting the list of all the documnets
    random.seed(42)
    centroids_list=list(random.sample(doc_list,k))#randomly picking k centroids from the doc_list
    for i in range(k):
        centroids[i]=centroids_list[i]
    #print(centroids)
    for iter in range(max_iter):
        for i in range(k):
                cluster[i] = []
        centroid_df = pd.DataFrame(jm).loc[centroids.values(),:] #taking only the rows of jaccard matrix correspoding to the centorid
        #print(centroid_df) 
        for cols in data:
            dist = list(centroid_df[cols-1])#taking the jaccard index of each document with respect to the centroids
            x = dist.index(min(dist))#finding the minimum
            cluster[x].append(cols-1)#add the doc to the cluster of the centroid corresponding to minimum distance
            #print(cluster)
        centroid_updated=[]
        for i in range(k):
            col_sums = list(pd.DataFrame(jm).loc[cluster[i],cluster[i]].sum(axis=0)) #finding the sum of jaccard distances for each document in a clsuter wrt to each other
            #print(col_sums)
            y = cluster[i][col_sums.index(min(col_sums))]#finding the lowest sum
            centroid_updated.append(y)#updating as the centroid
        for i in range(k):
            if iter!=max_iter-1: #last step
                centroids[i] = centroid_updated[i] #updating the new centroid in the centroid dictionary
    output={}
    output[0]=cluster
    output[1]=centroids
    return(output)

def kmean_inertia(data, jm,k=5,max_iter=10):
        cluster=kmeans(data,jm,k,max_iter)[0]
        centroids=kmeans(data,jm,k,max_iter)[1]
        x = 0
        for i in range(k):
            cen = centroids[i]
            clus = cluster[i]
            x += sum(jm[clus,cen]**2)
        return (x/k)
                
def optimize(data,jm,m,max_iter=10):
    x = []
    for i in range(1,m):
        x.append(kmean_inertia(data,jm,k=i,max_iter=10))
    return x




data1=data_from_txt("docword.enron.txt")        
data2=jaccard(data1)
res_nips=kmeans(data1,data2,3,8)
y=optimize(data1,data2,10)
plt.plot(list(range(1,10)),y)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.savefig('nips.png')
plt.show()

