
import numpy as np
import sklearn.metrics.pairwise

def assign_by_euclidian_at_k(X, T, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    distances = sklearn.metrics.pairwise.pairwise_distances(X)
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1] 
    return np.array([[T[i] for i in ii] for ii in indices])


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))


