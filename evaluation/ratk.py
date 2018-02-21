
import numpy as np
import sklearn.metrics.pairwise

def assign_by_euclidian_at_k(xs, ys, k):
    """ 
    xs : [sz_batch x nb_features], e.g. 100 x 64
    k : for each sample, assign labels of k nearest points
    """
    distances = sklearn.metrics.pairwise.pairwise_distances(xs)
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1] # get nearest points
    return np.array([[ys[i] for i in ii] for ii in indices])


def recall_at_k(ys, ys_pred_k, k):
    """
    ys : [sz_batch]
    ys_pred_k : [sz_batch x k]
    """
    s = sum([1 for y, y_pred_k in zip(ys, ys_pred_k) if y in y_pred_k[:k]])
    return s / (1. * len(ys))


