import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import os
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import argparse
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
import tifffile
import warnings
import sys
from skimage import exposure
from scipy import ndimage
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils
from pkg.SA import SA
from pkg.plot import scatterplot_3D, scatterplot_2D, construct_spot_image, plot_img_heatmap, get_spec_img
from pkg.clustering import kmeans_clustering, HDBSCAN_clustering, gaussian_mixture, hierarchical_clustering

warnings.filterwarnings('ignore', module='pyimzml')


def dimensionality_reduction(general_dict, umap_dict):
    """
    Computes dimensionality reduction on a MSI dataset of imzML format.

    PCA, t-sne and UMAP (semi-supervised or unsupervised) may be used for dimensionality reduction

    :param general_dict: a dictionary containing parameters for dimensionality reduction
    :param umap_dict: a dictionary containing parameters for UMAP
    """
    filename = os.path.basename(general_dict.get('imzML_file')).split('.')[0]
    p = ImzMLParser(general_dict.get('imzML_file'))

    # read spectral data
    spectra, coords, _ = utils.get_spectra_coords_arrays(general_dict.get('imzML_file'))
    spectra = np.nan_to_num(spectra)
    #spectra[spectra == 'nan'] = 0

    if general_dict['preembedding'] and spectra.shape[1] > 1000:
        print("TSVD embedding...")
        lsa = TruncatedSVD(n_components=1000, random_state=42)
        spectra = lsa.fit_transform(spectra)

    # compute dimensionality reduction
    if general_dict['method'] == 't-sne':
        print("t-sne embedding...")
        reducer = TSNE(n_components=general_dict['n_components'])
        embedding = reducer.fit_transform(spectra)
    elif general_dict['method'] == 'pca':
        print("PCA embedding...")
        reducer = PCA(n_components=general_dict['n_components'], svd_solver='full')
        embedding = reducer.fit_transform(spectra)
    else:
        # use learned umap embedding model or create new model
        if umap_dict['model_file'] != '':
            loaded_model = pickle.load(open(umap_dict['model_file'], 'rb'))
            embedding = loaded_model.transform(spectra)
        else:
            print("UMAP embedding...")
            labels = None
            # get labels for semi-supervised UMAP
            if umap_dict['supervised_dir'] != '':
                labels = np.full((len(p.coordinates)), -1, dtype=int)
                imzML_files = [f for f in os.listdir(umap_dict['supervised_dir']) if not f.startswith('.') and
                               f.endswith('.imzML') and os.path.isfile(os.path.join(umap_dict['supervised_dir'], f))]
                # get coordinates of all files
                for file_num, file in enumerate(imzML_files):
                    print("extracting coordinates from", file)
                    matches = 0
                    file_p = ImzMLParser(os.path.join(umap_dict['supervised_dir'], file))
                    file_coords = []
                    for (x, y, z) in tqdm(file_p.coordinates):
                        file_coords.append([x, y])
                    file_coords = np.array(file_coords)
                    for coord in file_coords:
                        index = np.where((coords == coord).all(axis=1))
                        if index[0].size > 0:
                            if labels[index[0]] == -1:
                                labels[index[0]] = file_num
                            matches += 1
                    print("found {} matching pixels for {}".format(matches, file))
                # plot image with labels
                im = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1, 3))
                for i, (x, y, z_) in enumerate(p.coordinates):
                    # print(labels[i])
                    im[y, x] = labels[i] + 2
                im = utils.NormalizeData(im)
                #plt.imshow(im, cmap='jet')
                plt.imshow(im, general_dict['cmap'])
                plt.axis('off')
                plt.savefig(os.path.join(general_dict['result_dir'], filename + '_' + general_dict['method'] +
                                         '_labeled_image.pdf'))
                if general_dict['debug']:
                    plt.show()
                plt.close()

            reducer = umap.UMAP(n_components=general_dict['n_components'], metric=umap_dict['metric'],
                                n_neighbors=umap_dict['n_neighbors'], min_dist=umap_dict['min_dist'],
                                verbose=general_dict['debug'])
            embedding = reducer.fit_transform(spectra, y=labels)
            pickle.dump(reducer, open(os.path.join(general_dict['result_dir'], filename + '_umap_model.sav'), 'wb'))

    embedding = utils.NormalizeData(embedding)

    # create scatter plot of embedding
    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
    scatterplot_output_file = os.path.join(general_dict['result_dir'], filename + '_scatterplot.png')
    if general_dict['n_components'] == 3:
        scatterplot_3D(data=embedding, filepath=scatterplot_output_file, c=embedding, cmap=general_dict['cmap'],
                       plot=general_dict['debug'], size=general_dict['dot_size'])
    elif general_dict['n_components'] == 2:
        scatterplot_2D(data=embedding, filepath=scatterplot_output_file,
                       c=None, cmap=general_dict['cmap'], plot=general_dict['debug'], size=general_dict['dot_size'])

    # construct hyperspectral visualization of embedding
    if general_dict['n_components'] == 1 or general_dict['n_components'] == 3:
        if general_dict['n_components'] == 1:
            embedding = embedding.flatten()
        umap_img = construct_spot_image(imzML_file=general_dict.get('imzML_file'), vals=embedding,
                                             output_file=os.path.join(general_dict['result_dir'],
                                                                      general_dict['method'] + '_'
                                                                      + str(general_dict['n_components']) + 'D_'
                                                                      + filename + '.svg'))
        umap_gray_img = umap_img
        if general_dict['n_components'] == 3:
            umap_gray_img = np.dot(umap_img[..., :3], [0.2989, 0.5870, 0.1140])
            norm_umap_gray_img = (utils.NormalizeData(exposure.equalize_hist(umap_gray_img))*255).astype('uint8')
        else:
            norm_umap_gray_img = (utils.NormalizeData(umap_gray_img) * 255).astype('uint8')
        tifffile.imwrite(os.path.join(general_dict['result_dir'],
                                      general_dict['method'] + '_grayscale_' + filename + '.tif'),
                         norm_umap_gray_img)
        plot_img_heatmap(umap_gray_img, os.path.join(general_dict['result_dir'],
                                                     general_dict['method'] + '_heatmap_' + filename + '.svg'))


    x = [i[0] for i in p.coordinates]
    y = [i[1] for i in p.coordinates]
    df_results = pd.DataFrame(embedding)
    df_results.insert(loc=0, column='x', value=x)
    df_results.insert(loc=1, column='y', value=y)

    return df_results


def segmentation(df, general_dict, cluster_dict):
    """
    Image segmentation via clustering of an MSI dataset.

    :param df: a pandas dataframe with columns x, y and m/z values or values of embedding from dim. reduction
    :param general_dict: a dictionary containing parameters for dimensionality reduction
    :param cluster_dict: a dictionary containing parameters for clustering
    """
    filename = os.path.basename(general_dict.get('imzML_file')).split('.')[0]
    msi_df, coords = utils.get_dataframe_from_imzML(general_dict.get('imzML_file'), multi_index=True, get_coords=True)
    p = ImzMLParser(general_dict.get('imzML_file'))
    pyx = (p.imzmldict["max count of pixels y"] + 1, p.imzmldict["max count of pixels x"] + 1)
    sp_data = df.iloc[:, 2:].to_numpy()    # spectral data

    meas_area_img = get_spec_img(pyx, coords)

    # perform clustering on embedding
    cluster_name = cluster_dict['cluster']
    if cluster_dict['cluster'] != '':
        if cluster_dict['cluster'] == 'k-means':
            class_labels = kmeans_clustering(data=sp_data, k=cluster_dict['n_clusters'])
            cluster_name = str(cluster_dict['n_clusters']) + '-means'
        elif cluster_dict['cluster'] == 'hdbscan':
            class_labels = HDBSCAN_clustering(data=sp_data, min_samples=cluster_dict['min_samples'],
                                              min_cluster_size=cluster_dict['min_cluster_size'], debug=general_dict['debug'])
            #print(np.unique(class_labels))
        elif cluster_dict['cluster'] == 'hierarchical':
            class_labels = hierarchical_clustering(data=sp_data, k=cluster_dict['n_clusters'], output_file=os.path.join(general_dict['result_dir'], filename + '_hdbscan.png'))
        elif cluster_dict['cluster'] == 'gaussian_mixture':
            class_labels = gaussian_mixture(data=sp_data, k=cluster_dict['n_clusters'])
        #elif cluster_dict['cluster'] == 'som':
        #    class_labels = som(data=sp_data, k=cluster_dict['n_clusters'])
        else:
            df_ = df.set_index(['x', 'y'])
            class_labels = SA(df_, k=cluster_dict['n_clusters'])
        df.insert(loc=2, column='class', value=class_labels)
        cluster_nos = np.unique(class_labels).tolist()

        # only keep matrix cluster in result file
        if cluster_dict['matrix_cluster']:
            # # matrix cluster has fewer values than tissue cluster if k=2
            # if cluster_dict['n_clusters'] == 2:
            #     matrix_class = df.groupby(['class']).size().idxmin()
            #     print('matrix class: ', matrix_class)
            # # matrix cluster contains most boundary pixels
            # else:
            # print(df)
            # create binary image of measured area
            meas_area_img = np.full((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1), 0).astype(np.uint8)
            for pixel, row in msi_df.iterrows():
                x, y = pixel[1], pixel[0]
                meas_area_img[x, y] = 1
            num_px_meas_area = np.count_nonzero(meas_area_img)
            tifffile.imwrite(os.path.join(general_dict['result_dir'], filename + '_tissue_img.tif'), (meas_area_img*255).astype('uint8'))
            
            # get boundary
            distance = ndimage.distance_transform_edt(meas_area_img)
            distance[distance != 1] = 0
            boundary = np.where(distance == 1)

            # create dataframe with columns x y of contours
            df_contours = pd.DataFrame.from_dict({'x': boundary[1], 'y': boundary[0]})
            # print(df_contours)

            # get number of contour points and number of pixels (percentage of total pixels) for each cluster
            # matrix_class = 0
            # contour_count = 0
            matrix_pixel_perc_thr = 100
            # dont consider cluster 0, since this is noise from hdbscan
            cluster_nos_without_zero = cluster_nos
            if 0 in cluster_nos:
                cluster_nos_without_zero.remove(0)
            print("clusters:")
            print(cluster_nos)
            # print(cluster_nos_without_zero)
            df_pixel_calc = pd.DataFrame(index=cluster_nos_without_zero, columns=['border pixels (%)', 'total pixels (%)'])
            for class_label in cluster_nos_without_zero:
                df_cluster = df[df['class'] == class_label]
                # get intersection of each cluster and count rows (=amount of contour points of cluster)
                contour_intersect = pd.merge(df_cluster, df_contours, on=['x', 'y'], how='inner')
                # print(contour_intersect.shape)
                # if contour_intersect.shape[0] > contour_count:
                #     contour_count = contour_intersect.shape[0]
                #     matrix_class = class_label
                # df_pixel_calc.loc[class_label, '# border pixels'] = contour_intersect.shape[0]
                df_pixel_calc.loc[class_label, 'border pixels (%)'] = (contour_intersect.shape[0] / df_contours.shape[0]) * 100
                df_pixel_calc.loc[class_label, 'total pixels (%)'] = (df_cluster.shape[0] / num_px_meas_area) * 100
            # print(df_pixel_calc)
            # keep clusters which have more than 0 border points
            df_pixel_calc = df_pixel_calc[df_pixel_calc['border pixels (%)'] > 0] 
            # print(df_pixel_calc)
            print('saving visualizations...')
            # print('cluster {} has {} contour pixels'.format(matrix_class, contour_count))

            # save visualization of pixel calculations
            fig, ax = plt.subplots()
            ax = df_pixel_calc.plot.bar(y='border pixels (%)', rot=0)
            plt.savefig(os.path.join(general_dict['result_dir'], filename + '_border_pixels.svg'))
            plt.close()
            fig, ax = plt.subplots()
            ax = df_pixel_calc.plot.bar(y='total pixels (%)', rot=0)
            ax.axhline(y=matrix_pixel_perc_thr, color= 'red', linewidth=5,)
            plt.savefig(os.path.join(general_dict['result_dir'], filename + '_total_pixels.svg'))
            plt.close()

            # get matrix class based on pixel calculations
            df_pixel_calc = df_pixel_calc[df_pixel_calc['total pixels (%)'] < matrix_pixel_perc_thr]
            # print(df_pixel_calc)
            idx_max_border_pts = np.argmax(df_pixel_calc['border pixels (%)'].to_numpy())
            print('index with max border pts=', idx_max_border_pts)
            idx = df_pixel_calc.index.to_numpy()
            matrix_class = idx[idx_max_border_pts]
            print('matrix class = ', matrix_class)

            # save dataframe with matrix pixels
            df_result = df[df['class'] == matrix_class].reset_index(drop=True)
            df_result['class'] = 1  # set class label to 1
            out = df_result.iloc[:, :2]
            #print(out)
        else:
            out = df

        # create 3D scatter plot of clusters
        if general_dict['method'] != '':
            scatterplot_cluster_output_file = os.path.join(general_dict['result_dir'], filename + '_cluster_scatterplot.png')
            if general_dict['n_components'] == 3:
                scatterplot_3D(data=sp_data, filepath=scatterplot_cluster_output_file, c=class_labels,
                               plot=general_dict['debug'], cmap=general_dict['cmap'], size=general_dict['dot_size'])
            if general_dict['n_components'] == 2:
                scatterplot_2D(data=sp_data, filepath=scatterplot_cluster_output_file, c=class_labels,
                               plot=general_dict['debug'], cmap=general_dict['cmap'], size=general_dict['dot_size'])

        # construct clustered image
        # clustered_gray_im = construct_spot_image(imzML_file=general_dict.get('imzML_file'), vals=class_labels,
        #                                     output_file=os.path.join(general_dict['result_dir'], filename + '_'
        #                                                              + general_dict['method'] + '_'
        #                                                              + cluster_dict[
        #                                                                  'cluster'] + '_clustered_image.tif'))
        # if cluster_dict['matrix_cluster']:
        #     cluster_output_file = ''
        # else:
        cluster_output_file = os.path.join(general_dict['result_dir'], filename + '_cluster_image.svg')
        clustered_im = construct_spot_image(imzML_file=general_dict.get('imzML_file'), vals=class_labels,
                                            output_file=cluster_output_file, cmap=general_dict['cmap'])
        tifffile.imwrite(os.path.join(general_dict['result_dir'], filename + '_cluster_image.tif'), clustered_im)
        # save binary images of clusters
        if cluster_dict['matrix_cluster']:
            class_im = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1))
            class_im[clustered_im == matrix_class] = 255
            im_file = os.path.join(cluster_dict['cluster_dir'], filename + '_matrix_cluster.tif')
            tifffile.imwrite(im_file, class_im.astype('uint8'))
            if general_dict['debug']:
                plt.imshow(class_im)
                plt.axis('off')
                plt.show()
                plt.close()
        #else: 
        for class_label in cluster_nos:
            # only save binary images of rest clusters (not matrix)
            if cluster_dict['matrix_cluster'] and class_label == matrix_class:
                pass
            else:
                class_im = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1))
                class_im[clustered_im == class_label] = 255
                if class_label == 0:
                    class_im[meas_area_img == 0] = 0
                im_file = os.path.join(cluster_dict['cluster_dir'], filename + '_cluster' + str(class_label) + '.tif')
                tifffile.imwrite(im_file, class_im.astype('uint8'))
                if general_dict['debug']:
                    plt.imshow(class_im)
                    plt.axis('off')
                    plt.show()
                    plt.close()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes dimensionality reduction and clustering on a MSI dataset')
    parser.add_argument('imzML_file', type=str, help='MSI imzML file')
    parser.add_argument('-supervised_dir', type=str, default='', help='directory to imzML files for semi-supervised UMAP')
    parser.add_argument('-result_dir', type=str, default='', help='directory to save the results, default \'\' will create a results folder')
    parser.add_argument('-method', type=str, default='umap', help='dimensionality reduction method, default \'umap\'')
    parser.add_argument('-n_components', type=int, default=2, help='desired number of dimensions, default=2')
    parser.add_argument('-metric', type=str, default='cosine', help='distance metric for umap, default=\'cosine\'')
    parser.add_argument('-n_neighbors', type=int, default=15, help='number of neighbors for umap, default=15')
    parser.add_argument('-model_file', type=str, default='', help='file of UMAP model, default=\'\'')
    parser.add_argument('-min_dist', type=float, default=0.1, help='min_dist for UMAP, default=0.1')
    parser.add_argument('-cluster', type=str, default='k-means', help='clustering algorithm, default=\'k-means\'')
    parser.add_argument('-n_clusters', type=int, default=5, help='number of clusters, default=5')
    parser.add_argument('-min_cluster_size', type=int, default=1000, help='min number of clusters for hdbscan, default=1000')
    parser.add_argument('-min_samples', type=int, default=5, help='min number of samples for hdbscan, default=5')
    parser.add_argument('-matrix_cluster', type=bool, default=False, help='set to True to extract only matrix pixels, default=False')
    parser.add_argument('-preembedding', type=bool, default=False, help='set to True to perform TSVD as initial'
                                                                        'dimensionality reduction, default=False')
    parser.add_argument('-cmap', type=str, default='Spectral', help='cmap to use for all plots, default=\'Spectral\'')
    parser.add_argument('-dot_size', type=int, default=1, help='size for dots in scatterplots, default=1')
    parser.add_argument('-debug', type=bool, default=False, help='set to true if output should be plotted, default=False')
    args = parser.parse_args()

    if args.supervised_dir != '':
        args.method = 'umap'
    if args.result_dir == '':
        if args.method != '':
            result_dir = os.path.join(os.path.dirname(args.imzML_file), args.method + '_' +
                                      str(args.n_components) + 'dim')
            if args.method == 'umap':
                result_dir += '_' + str(args.n_neighbors) + 'neighb'

            if args.cluster != '':
                if args.cluster == 'hdbscan':
                    result_dir += '_' + args.cluster + '_' + str(args.min_samples) + 'ms_' + str(args.min_cluster_size) + 'mcs'
                elif args.cluster == 'k-means':
                    result_dir += '_' + str(args.n_clusters) + '-means'

        else:
            result_dir = os.path.join(os.path.dirname(args.imzML_file), args.cluster)
        args.result_dir = result_dir
    # if not os.path.exists(args.result_dir):
    #     os.mkdir(args.result_dir)
    try:
        os.mkdir(args.result_dir)
    except OSError:
        pass
    if args.cluster != '':
        # if args.matrix_cluster:
        #     cluster_dir = args.result_dir
        # else:
        cluster_dir = os.path.join(args.result_dir, 'binary_imgs')
        #if not os.path.exists(os.path.abspath(cluster_dir)):
        #    os.mkdir(os.path.abspath(cluster_dir))
        try:
            os.mkdir(cluster_dir)
        except OSError:
            pass

    else:
        cluster_dir = ''

    general_params = {
        "imzML_file": args.imzML_file,
        "result_dir": args.result_dir,
        "method": args.method,
        "n_components": args.n_components,
        "preembedding": args.preembedding,
        "cmap": args.cmap,
        "dot_size": args.dot_size,
        "debug": args.debug
    }

    umap_params = {
        "metric": args.metric,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "model_file": args.model_file,
        "supervised_dir": args.supervised_dir
    }

    cluster_params = {
        "cluster": args.cluster,
        "n_clusters": args.n_clusters,
        "matrix_cluster": args.matrix_cluster,
        "cluster_dir": cluster_dir,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples
    }
    cluster_name = cluster_params['cluster']
    if cluster_params['cluster'] == 'k-means':
        cluster_name = str(cluster_params['n_clusters']) + cluster_params['cluster']

    filename = os.path.basename(general_params.get('imzML_file')).split('.')[0]

    if general_params['method'] != '':
        df = dimensionality_reduction(general_params, umap_params)
    else:
        df = utils.get_dataframe_from_imzML(general_params['imzML_file'], multi_index=False)
        # df = utils.imzml_to_df(general_params['imzML_file'])
        # df = df.droplevel('dataset')
        # df.index.set_names(["x", "y"], inplace=True)

    if cluster_params['cluster'] != '':
        clustered = segmentation(df, general_params, cluster_params)

        # save pixels, embedding and class labels as csv
        out_file = ''
        if cluster_params['matrix_cluster']:
            out_file = os.path.join(general_params['result_dir'], filename + '_matrix_pixels.csv')
        else:
            if general_params['method'] != '':
                out_file = os.path.join(general_params['result_dir'], filename + '_' + general_params['method'] + '_' +
                                   str(general_params['n_components']) + 'D_embedding_' + cluster_name + '_clustering.csv')
            else:
                out_file = os.path.join(general_params['result_dir'], filename + '_' + cluster_name + '_clustering.csv')
        print("saving output to " + out_file)
        clustered.to_csv(out_file, index=True)
        print("done")

