import cv2
import hdbscan
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import AgglomerativeClustering
# from sklearn_som.som import SOM


#def som(data, k):
#    print(data.shape)
#    som = SOM(m=k, n=1, dim=data.shape[1])
#    som.fit(data)
#
#    class_labels = som.predict(data)
#    class_labels = class_labels.ravel() + 1
#    return class_labels


def hierarchical_clustering(data, k):
    model = AgglomerativeClustering(n_clusters=k)
    class_labels = model.fit_predict(data)
    return class_labels


def hierarchical_clustering_sk(data, connectivity=None):
    print('\tHCA')
    return AgglomerativeClustering(distance_threshold=0, n_clusters=None, connectivity=connectivity).fit(data)


def gaussian_mixture(data, k, start=1):
    print('Gaussian mixture model...')
    model = GaussianMixture(n_components=k)
    model.fit(data)
    class_labels = model.predict(data)
    return class_labels + start


def kmeans_clustering(data, k):
    """
    Performs k-means clustering and returns class labels.

    :param data: data to cluster
    :param k: number of clusters
    :type data: numpy array of shape (m, n)
    :type k: int
    :return: class labels
    :rtype: numpy array of shape (m,)
    """
    print('k-means clustering...')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, labels, center = cv2.kmeans(data=data.astype(np.float32), K=k, bestLabels=None, criteria=criteria, attempts=10,
                                     flags=cv2.KMEANS_RANDOM_CENTERS)
    class_labels = labels.ravel() + 1  # change so first cluster is 1
    return class_labels


def HDBSCAN_clustering(data, min_samples=5, min_cluster_size=5, start=1, cmap='Spectral', debug=False, output_file=''):
    """
    Performs HDBSCAN clustering and returns class labels.

    :param data: data to cluster
    :param debug: if True scatter plot with clusters is plotted
    :param output_file: file path to store figure
    :type data: numpy array of shape (m, n)
    :type debug: bool
    :type output_file: str
    :return: class labels
    :rtype: numpy array of shape (m,)
    """
    print('HDBSCAN clustering...')
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(data)
    class_labels = labels + start
    clustered = (labels > start-1)

    if data.shape[1] == 2:
        plt.scatter(data[~clustered, 0], data[~clustered, 1], c=(0.5, 0.5, 0.5), s=0.1, alpha=0.5)
        if debug:
            plt.show()

        plt.scatter(data[clustered, 0], data[clustered, 1], c=labels[clustered], s=1, cmap=cmap)
        plt.title('HDBSCAN clustering')
        if debug:
            plt.show()
        if output_file != '':
            plt.savefig(output_file)
        plt.close()

    return class_labels
