# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import warnings

try:
    from sklearn.cluster import KMeans as sk_KMeans
except ImportError:
    sk_KMeans = None

try:
    from sklearn.cluster import MiniBatchKMeans as sk_MiniBatchKMeans
except ImportError:
    sk_MiniBatchKMeans = None

try:
    from sklearn.decomposition import TruncatedSVD as sk_TruncatedSVD
except ImportError:
    sk_TruncatedSVD = None


def kmeans(x_trains, n_clusters, batch_version, **kmeans_params):
    def fit(x_, **kmeans_params_):
        if batch_version is False:
            try:
                model = sk_KMeans(**kmeans_params_).fit(x_)
            except MemoryError:
                warnings.warn("An MemoryError occurred during the execution of k-Means non-batch version. Be careful "
                              "that all provides parameters via `kmeans_params` are accept by the batch version and "
                              "have the desired value.")
                model = sk_MiniBatchKMeans(**kmeans_params_).fit(x_)
        else:
            model = sk_MiniBatchKMeans(**kmeans_params_).fit(x_)

        return model.cluster_centers_, model.labels_

    if sk_KMeans is None or sk_MiniBatchKMeans is None:
        raise ImportError('`pre_training` requires sklearn.')

    if not isinstance(n_clusters, (list, tuple)):
        n_clusters = np.repeat(n_clusters, len(x_trains))
    elif len(n_clusters) != len(x_trains):
        raise TypeError("n_clusters not understood. Provide n_clusters as int or list with len(x_trains).")

    labels = []
    clusters = []
    for i, x in enumerate(x_trains):
        kmeans_params.update({'n_clusters': n_clusters[i]})
        cluster, label = fit(x, **kmeans_params)
        clusters.append(cluster)
        labels.append(label)

    return clusters, labels


def svd(clusters,
        n_components,
        **svd_params):
    if sk_TruncatedSVD is None:
        raise ImportError('`pre_training` requires sklearn.')

    if not isinstance(n_components, (list, tuple)):
        n_components = np.repeat(n_components, len(clusters))
    elif len(n_components) != len(clusters):
        raise TypeError("n_components not understood. Provide n_components as int or list with len(clusters).")

    matrices = []
    for i, c in enumerate(clusters):
        if n_components[i] == c.shape[-1]:
            matrix = np.linalg.eig(np.matmul(c.transpose(), c))[1]
        else:
            svd_params.update({'n_components': n_components[i]})
            model = sk_TruncatedSVD(**svd_params)
            model.fit(c)
            matrix = model.components_.transpose()
        matrices.append(matrix)

    return matrices
