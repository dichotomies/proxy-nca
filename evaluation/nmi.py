import sklearn.cluster
import sklearn.metrics.cluster

def cluster_by_kmeans(xs, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    return sklearn.cluster.KMeans(nb_clusters).fit(xs).labels_

def calc_nmi(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)
