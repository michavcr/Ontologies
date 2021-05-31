#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:48:46 2021

@author: michael
"""
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np

def visualize_UMAP(embeddings,target, w=12, h=6):
    reducer = umap.UMAP()

    data = reducer.fit_transform(embeddings)
    plt.figure(figsize=(w, h))
    
    plt.title("UMAP visualization of the embeddings")
    plt.scatter(data[:,0],data[:,1],c=target, cmap='jet')
    
    n = target.max() + 1
    
    centers = np.zeros((n, 2))
    incr = np.zeros((n, 1))
    num = np.arange(0,n)
    
    for i, t in enumerate(target):
        centers[t,:] += data[i,:]
        incr[t,:] += 1
    
    centers = centers / incr
    
    plt.scatter(centers[:,0], centers[:,1], c = 'black')
    
    for i, txt in enumerate(num):
        plt.annotate(txt, (centers[i,0], centers[i,1]))

    return

def visualize_TSNE(embeddings,target):
    tsne = TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=30)
    data = tsne.fit_transform(embeddings)
    #plt.figure(figsize=(12, 6))
    plt.title("TSNE visualization of the embeddings")
    plt.scatter(data[:,0],data[:,1],c=target)

    return

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def quantile_columns(M, mask, p=0.9):
    X=np.quantile(M, 0.70, axis=0)
    
    return(M[:,X>0], mask[X>0,:])