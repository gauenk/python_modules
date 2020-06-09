import numpy as np
import numpy.random as npr
from poisson_process import PoissonProcess

class CRP():
    
    def __init__(self,alpha=30,n=None,c=None):
        # init
        self.alpha = alpha
        self.n = 0
        self.c = []
        self.sizes = []

        # error checking
        if n is not None:
            if len(c) != n:
                raise InputError("Number of clusters must be equal to cluster assigment length")
            self.n = n
            self.c = c
            self.sizes = self.bincount(c)

    def sample(self,n=1):
        clusters = []
        for i in range(n):
            cluster = self.single_sample()
            clusters.append(cluster)
        return clusters

    def single_sample(self):
        denom = self.n - 1 + self.alpha
        if denom > 0:
            prob = self.alpha / denom
        else:
            prob = 1.0
        # coin flip
        acc = npr.uniform(0,1) < prob
        if acc or self.n == 0: # start a new table
            self.sizes += [0]
            cluster = len(self.sizes) - 1 # yes "n + 1" but "-1" since 0 indexed
        else: # join an old one.
            probs = self.sizes / np.sum(self.sizes)
            cluster = np.where(npr.multinomial(1,probs))[0][0]
        self.sizes[cluster] += 1
        self.n += 1
        self.c += [cluster]
        return cluster


