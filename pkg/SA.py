import math
from random import *
import numpy as np
from math import sqrt, pow
import os
import pandas as pd

from pkg.utils import get_dataframe_from_imzML
from pkg.clustering import hierarchical_clustering_sk, kmeans_clustering
from pkg.plot import construct_spot_image


def choose_distant_objects(col):
    """
    Choose the pivot objects Oa and Ob for which the distance is maximized to create a line on which objects can
    be projected.
    :return: the desired pair of pivot objects with the maximal distance.
    """
    global X

    # 1 choose arbitrarily an object, and declare it to be the second pivot object obj_b
    obj_b = randint(0, X.shape[0] - 1)

    # 2 set obj_a = object that is farthest apart from obj_b according to distance function
    # calculate distances from Ob to all other objects
    dists_to_obj_b = np.empty((X.shape[0]))
    for i in range(X.shape[0]):
        dists_to_obj_b[i] = distance_between_projections(i, obj_b, col)
    # select object with maxiumum distance as obj_b
    obj_a = np.argmax(dists_to_obj_b)

    # 3 set obj_b = object that is farthest apart from obj_a
    dists_to_obj_a = np.empty((X.shape[0]))
    for i in range(X.shape[0]):
        dists_to_obj_a[i] = distance_between_projections(i, obj_a, col)
    # select object with maxiumum distance as obj_b
    obj_b = np.argmax(dists_to_obj_a)

    # report objects Oa and Ob as the desired pair of objects
    return obj_a, obj_b


def distance_between_projections(i, j, col):
    """
    :param i: first object id
    :param j: second object id
    :return: distance between Oi and Oj according to equation 4 of Fastmap
    """
    global data_matrix
    global X

    if col == 0:
        return np.linalg.norm(data_matrix[j]-data_matrix[i])
    else:
        res = math.pow(distance_between_projections(i, j, col - 1), 2) - (math.pow(X[i][col - 1] - X[j][col - 1], 2))
        return math.sqrt(np.abs(res))


def Fastmap(k):
    """
    Find N points in k-d space whose euclidean distances match the distances of a given NxN distance matrix.
    Projection of objects as points in a n-dimensional space on k mutually directions.
    :param k: desired number of dimensions
    """
    global data_matrix  # original input data
    global X  # output with the ith row being the image of the ith object at the end of the algorithm
    global PA  # stores the ids of the pivot objects - one pair per recursive call
    global col  # stores the column of the X array currently being updated

    print("k=", k)
    # 1
    if k <= 0:
        return
    else:
        col += 1

    # 2 choose pivot objects
    obj_a, obj_b = choose_distant_objects(col)

    # 3 record the ids of the pivot objects
    PA[0, col] = obj_a
    PA[1, col] = obj_b

    # print(PA)

    # 4 if distance between pivot objects is 0, set X[i, col] = 0 because all inter-object distances are 0
    if distance_between_projections(obj_a, obj_b, col) == 0:
        for i in range(X.shape[0]):
            X[i, col] = 0
    # 5 project the objects on the line (obj_a, obj_b) for each obj_i according equation 3
    else:
        for i in range(X.shape[0]):
            # projection of first pivot is always 0
            if i == obj_a:
                X[i, col] = 0
            # projection of second pivot is always its distance from the first
            elif i == obj_b:
                X[i, col] = distance_between_projections(obj_a, obj_b, col)
                # print("dist=", distance_between_projections(obj_a, obj_b, col))
            else:
                X[i, col] = (pow(distance_between_projections(obj_a, i, col), 2) + pow(
                    distance_between_projections(obj_a, obj_b, col), 2) - pow(
                    distance_between_projections(obj_b, i, col), 2)) / (
                                        2 * distance_between_projections(obj_a, obj_b, col))
    # 6 recursive call
    Fastmap(k - 1)


def get_weight_matrix(r):
    # 1 create list with indices ranging from -r to r
    ind = list(range(-r, r + 1, 1))

    # 2 define an empty dataframe with with index from r to -r and columns from -r to r
    weight_matrix = pd.DataFrame(columns=ind, index=ind[::-1])

    # 3 in a for loop iterate through all indices and fill weight matrix dataframe values with equation 2
    for i in ind:
        for j in ind:
            sigma = (2 * r + 1) / 4
            weight_matrix.at[i, j] = math.exp((-pow(i, 2) - pow(j, 2)) / (2 * pow(sigma, 2)))

    # 4 return weight matrix
    return weight_matrix


def SA(msi_df, r=3, q=20, k=10, connectivity=None, seed_val=0, cluster_mode='kmeans'):
    """
    Spatially-aware clustering

    :param msi_df: pandas dataframe with (x,y) as multi index and m/z values as columns
    :param r: pixel neighborhood radius
    :param q: Fastmap desired dimension
    :param k: number of clusters
    :param connectivity: include pixel connectivity information during clustering
    :param seed_val: seed the random number generator
    :param cluster_mode: whether to use kmeans or HCA
    :return: class labels
    """

    global data_matrix  # original input data
    global X  # output with the ith row being the image of the ith object at the end of the algorithm
    global PA  # stores the ids of the pivot objects - one pair per recursive call
    global col  # stores the column of the X array currently being updated

    no_pixels = msi_df.shape[0]
    no_peaks = msi_df.shape[1]

    # #######################
    # # 1. given r, create weights
    # #######################
    weight_matrix_df = get_weight_matrix(r)

    # #######################
    # # 2. map a spectrum into the feature space
    # #######################
    # create empty numpy data array of shape no_pixels x weight_matrix_rows x weight_matrix_cols x no_peaks
    data = np.empty((no_pixels, weight_matrix_df.shape[0], weight_matrix_df.shape[1], no_peaks))

    # list with indices ranging from -r to r
    ind = list(range(-r, r + 1, 1))

    # fill data array by iteratively looping through spectra
    # set pixel counter
    px_idx = 0
    for pixel, spectra in msi_df.iterrows():
        # create an empty array to save phi(s)
        phi_s = np.empty((weight_matrix_df.shape[0], weight_matrix_df.shape[1], no_peaks))

        # iterate through every index in weight matrix and fill phi(s)
        for i in ind:
            for j in ind:
                # set x and y pixel according to i and j
                x = i + pixel[0]
                y = j + pixel[1]

                # if x and y in dataframe store spectra at x,y in s, otherwise set s as zero vector
                if (x, y) in msi_df.index:
                    s = msi_df.loc[(x, y)].to_numpy()
                else:
                    s = np.zeros(no_peaks)

                # calculate phi
                phi_s[i, j] = math.sqrt(weight_matrix_df.at[i, j]) * s

        # insert phi(s) into data matrix
        data[px_idx] = phi_s

        # increase pixel counter
        px_idx += 1

    #######################
    # 3. Given q, project mapped spectra into Rq using FastMap
    #######################
    data_matrix = data
    X = np.zeros((data_matrix.shape[0], q))
    PA = np.zeros((2, q))
    col = -1

    seed(seed_val)
    Fastmap(q)

    np.set_printoptions(suppress=True)
    # print(X)
    # print(PA)
    # print(X.shape)

    #######################
    # 4. Cluster the projected mapped spectra into k groups using k-means
    #######################
    if cluster_mode == 'kmeans':
        class_labels = kmeans_clustering(data=np.float32(X), k=k)
    else:
        class_labels = hierarchical_clustering_sk(data=np.float32(X), connectivity=connectivity)

    return class_labels


if __name__ == '__main__':
    imzML_fl = '/Users/philippaspangenberg/Desktop/bladder4/test/low_res/imzML/blase 4.2 - total ion count - reduced.h5/blase 4.2 - total ion count - reduced_autopicked/autopicked.imzML'
    imzML_dir = os.path.dirname(imzML_fl)
    msi_df = get_dataframe_from_imzML(imzML_fl)

    class_labels = SA(msi_df)

    # construct clustered image
    clustered_im = construct_spot_image(imzML_file=imzML_fl, vals=class_labels,
                                        output_file=os.path.join(os.path.join(imzML_dir, 'fastmap_kmeans.svg')))