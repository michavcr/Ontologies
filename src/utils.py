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

def visualize_UMAP(embeddings, target, w=12, h=6):
    """
    Allows to visualize vectors (embeddings from a neural network model) in a 
    2d-space, using the UMAP algorithm.
    Parameters
    ----------
    embeddings : a numpy ndarray of shape (n_samples, embd_size)
        The vectors to plot in the 2D graph.
    target : a numpy ndarray of shape (n_samples,)
        The targets associated with each embedding vector. Should be integers.
    w : int or float, optional
        The weight of the figure. The default is 12.
    h : int or float, optional
        The height of the figure. The default is 6.

    Returns
    -------
    None.

    """
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

def visualize_TSNE(embeddings, target, w=12, h=6):
    """
    Allows to visualize vectors (e.g. embeddings from a neural network model) 
    in a 2d-space, using the t-SNE algorithm.
    Parameters
    ----------
    embeddings : a numpy ndarray of shape (n_samples, embd_size)
        The vectors to plot in the 2D graph.
    target : a numpy ndarray of shape (n_samples,)
        The targets associated with each embedding vector. Should be integers.
    w : int or float, optional
        The weight of the figure. The default is 12.
    h : int or float, optional
        The height of the figure. The default is 6.

    Returns
    -------
    None.

    """
    tsne = TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=30)
    data = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(w, h))
    plt.title("TSNE visualization of the embeddings")
    plt.scatter(data[:,0],data[:,1],c=target)

    return

def timeit(method):
    """
    Time decorator to print the execution time of a function.

    Use @timeit before your functions.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    
    """
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

def quantile_columns(M, mask, p=0.70):
    X=np.quantile(M, p, axis=0)
    
    return(M[:,X>0], mask[X>0,:])