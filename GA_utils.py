#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


### binary to real value:
def chrome_decode(Population, BS, m, Lo, Hi):
    N = Population.shape[0]
    #real_val=np.zeros((0))
    real_val = np.zeros((N, m), dtype=np.float32) ## real values Dims : N by m; for accuracy: float64 or float32 
    # kernels are m array filled by powers of 2 for convert between binary to real: for e.g.: [[1, 2, 4, 8, ...], [1, 2, 4, 8,...], ...]
    kernels = np.array([[2**i for i in range(BS[j])] for j in range(BS.shape[0])])
    for n in range(N):
        start_point = 0
        for i in range(m):
            stop_point = start_point + BS[i]
            a = Population[n, start_point:stop_point]
            # calculate the real value and normalizing:
            r = np.array(np.sum(a.dot(kernels[i,:])))  / (2 ** (BS[i]) - 1) 
            r = r * (Hi[i] - Lo[i]) + Lo[i] # map to Hi and Lo
            real_val[n, i] = r
            start_point = stop_point
    return real_val


### cost function:
def cost_function(X):
    ''' This is a Cost function and you can change it on your purposes!'''
    return (1 + np.cos(2 * np.pi * X[:,0] * X[:, 1])) * np.e ** (-(abs(X[:,0] + X[:,1])/2))


### Selection:
def selection(fitness, X):
    P = fitness / np.sum(fitness)
    CDF = np.cumsum(P)
    CDF = CDF.tolist()
    CDF.insert(0, 0) # put zero at start point of CDF, in numpy it is diffcult and time consuming to change size an array!
    CDF = np.array(CDF)
    
    # Roullete Weel:
    RW = np.random.mtrand.uniform(size=(1, P.shape[0]))[0]
    idx = np.zeros((1, P.shape[0]), np.int)[0]
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if CDF[j] <= RW[i] < CDF[j+1]:
                idx[i] = j
    # Selection:
    Parents = np.zeros_like(X)
    for i in range(len(idx)):
        Parents[i,:] = X[idx[i], :].copy()  # I don't no copy is necessary or not!
    return Parents


### Cross over:
def crossingover(Parents, Pc):
    N, L = Parents.shape
    children = np.zeros((N, L))
    isCross = np.random.mtrand.uniform(size=(1, int(N / 2)))[0] < Pc
    i = 0
    j = 0
    while i <= N - 1:
        P1 = Parents[i,:].tolist()
        P2 = Parents[i+1,:].tolist()
        if isCross[j]:
            c = np.floor(np.random.mtrand.uniform(L))
            c = int(c)
            child1 = np.array(P1[0:c]+ P2[c:])
            child2 = np.array(P2[0:c]+ P1[c:])
        else:
            child1 = np.array(P1)
            child2 = np.array(P2)
        children[i][:] = child1.copy()
        children[i+1][:] = child2.copy()
        i += 2
        j+= 1
    return children


### Mutation:
def mutation(X, Pm):
    N, L = X.shape
    mutate = np.random.mtrand.uniform(size=(N, L)) < Pm
    Xm = np.bitwise_xor(X.astype(np.int), mutate);
    return Xm
