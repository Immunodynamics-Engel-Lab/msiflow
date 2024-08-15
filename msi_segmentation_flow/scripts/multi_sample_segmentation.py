import numpy as np
import argparse
import os
from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import plotly.express as px
# import hvplot
# import hvplot.pandas
from sklearn.manifold import TSNE
import warnings
import sys
import tifffile
from sklearn.decomposition import PCA, TruncatedSVD
import pickle
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils
from pkg.clustering import kmeans_clustering, HDBSCAN_clustering, hierarchical_clustering, gaussian_mixture

warnings.filterwarnings('ignore', module='pyimzml')

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})


def plot_clustered_imgs(file_dir, file_list, df_data, result_dir):
    """
    Creates hyperspectral visualization of imzML files of a specific group according to cluster labels and saves
    images as png files.

    :param file_dir: directory of imzML files of a specific group
    :param file_list: list of files, e.g. ['1-heart.imzML', '2-heart.imzML']
    :param df_data: data frame containing columns 'group' 'sample' 'x' 'y' 'UMAP_1' 'UMAP_2' 'label' 'label_color'
    :param result_dir: directory to store results
    :type file_dir: str
    :type file_list: list[str]
    :type df_data: pandas.DataFrame
    :type result_dir: str

    .. warning:: files must be named like "1-heart.imzML" starting with the sample number followed by "-".
    """
    for img_fl in file_list:
        # animal = img_fl.split('.')[0][-1]
        #animal = img_fl.split('.')[0]
        #animal = img_fl.split('.')[0].split('_')[1]
        #sample = (img_fl.split('.')[0]).split('_')[1][-2:]
        sample = (img_fl.split('.')[0])
        is_sample = df_data['sample'] == sample
        df_sample_result = df_data[is_sample]
        p = ImzMLParser(os.path.join(file_dir, img_fl))
        im = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1, 4))

        x_coords = df_sample_result['x'].to_numpy()
        y_coords = df_sample_result['y'].to_numpy()
        labels = df_sample_result['label_color'].to_list()

        # if label with -1 (noise) set label color to white
        # df_sample_result.loc[df_sample_result['label'] == -1, 'label_color'] = tuple(0.5, 0.5, 0.5, 0.5)
        #print(df_sample_result[df_sample_result['label'] == -1])

        z = np.tile(np.array([0, 1, 2, 3]), x_coords.shape)
        x = np.repeat(x_coords, 4)
        y = np.repeat(y_coords, 4)

        labels = [item for t in labels for item in t]
        labels = np.asarray(labels)
        im[y, x, z] = labels
        # for i, (x, y, z_) in enumerate(p.coordinates):
        #     label = df_sample_result.loc[(df_sample_result['x'] == x) & (df_sample_result['y'] == y), 'label'].iloc[0]
        #     if label == -1:  # set unlabeled pixel to white
        #         label_color = (0.0, 0.0, 0.0, 00)
        #     else:
        #         label_color = df_sample_result.loc[(df_sample_result['x'] == x) & (df_sample_result['y'] == y),
        #                                            'label_color'].iloc[0]
        #     im[y, x, 0] = label_color[0]
        #     im[y, x, 1] = label_color[1]
        #     im[y, x, 2] = label_color[2]
        #     im[y, x, 3] = label_color[3]
        plt.imshow(im)
        plt.axis('off')
        plt.imsave(os.path.join(result_dir, img_fl.split('.')[0] + '.png'), im, dpi=300)
        plt.close()


def extract_binary_imgs(file_dir, file_list, df_data, result_dir):
    for img_fl in file_list:
        sample = (img_fl.split('.')[0])
        is_sample = df_data['sample'] == sample
        df_sample_result = df_data[is_sample]
        labels = df_sample_result['label']
        p = ImzMLParser(os.path.join(file_dir, img_fl))

        for class_label in range(labels.min(), labels.max()+1):
            im = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1))
            # for i, (x, y, z_) in enumerate(p.coordinates):
            #     label = df_sample_result.loc[(df_sample_result['x'] == x) & (df_sample_result['y'] == y), 'label'].iloc[
            #         0]
            #     if label == class_label:
            #         im[y, x] = 255
            df_class = df_sample_result[df_sample_result['label'] == class_label]
            x_coords = df_class['x'].to_numpy()
            y_coords = df_class['y'].to_numpy()
            im[y_coords, x_coords] = 255

            plt.imshow(im)
            plt.axis('off')
            im_file = os.path.join(result_dir, str(class_label) + '_' + sample + '.tif')
            tifffile.imwrite(im_file, im.astype('uint8'))
            plt.close()


def umap_groups(imzml_dir, result_dir, clustering='HDBSCAN', min_clusters=50, min_samples=5, method='umap',
                dist_metric='cosine', n_neighbors=50, min_dist=0.1, n_clusters=10, preembedding=False,
                preembedding_model='', embedding_model='', dot_size=2, cmap='Spectral'):
    """
    Computes dimensionality reduction (UMAP or t-sne) on multiple imzML files (which must have a common m/z vector)
    of different groups. UMAP/t-sne and clustering result is saved as csv file. Also scatter plots of UMAP/t-sne,
    clusters and hyperspectral images of the imzML files according to the clustering are generated.

    :param imzml_dir: directory containing imzML files of corresponding groups
    :param result_dir: directory to store results
    :param min_clusters: minimal cluster size for HDBSCAN clustering
    :param method: dimensionality method (either 'umap' or 't-sne')
    :type groups_dir: str
    :type result_dir: str
    :type min_clusters: int
    :type method: str
    """
    # Get combined data frame of all groups
    # group_files_dict = {}
    # df_combined = pd.DataFrame()
    # for group in groups_folder_list:
    #     group_file_path = os.path.join(groups_dir, group)
    #     group_files_dict[group] = [f for f in os.listdir(group_file_path) if f.endswith('.imzML')
    #                                and not f.startswith('.') and os.path.isfile(os.path.join(group_file_path, f))]
    #     df = utils.get_combined_dataframe_from_group_files(group_file_path, group_files_dict[group])
    #     df_combined = df_combined.append(df, ignore_index=True)
    #     print("found {} files for {}".format(len(group_files_dict[group]), group))
    file_list = [f for f in os.listdir(imzml_dir) if f.endswith('.imzML')]
    df_combined = utils.get_combined_dataframe_from_files(imzml_dir, file_list, groups=True)
    df_meta_data = df_combined.iloc[:, :4]
    spec_data = df_combined.iloc[:,4:].to_numpy()
    # print(df_combined)
    if preembedding and spec_data.shape[1] > 1000:
        print("TSVD embedding started at {}".format(datetime.now()))
        if preembedding_model != '':
            lsa = pickle.load(open(preembedding_model, 'rb'))
        else:
            lsa = TruncatedSVD(n_components=1000, random_state=42).fit(spec_data)
            pickle.dump(lsa, open(os.path.join(result_dir,"tsvd_model.sav"), 'wb'))
        spec_data = lsa.transform(spec_data)
        print("TSVD finished at {}".format(datetime.now()))

    # 2D-embedding
    print('{} started at {}'.format(method, datetime.now()))
    if method == 't-sne':
        reducer = TSNE(n_components=2)
        embedding = reducer.fit_transform(spec_data)
    else:
        if embedding_model != '':
            reducer = pickle.load(open(embedding_model, 'rb'))
        else:
            reducer = umap.UMAP(n_components=2, metric=dist_metric, n_neighbors=n_neighbors, min_dist=min_dist, random_state=None).fit(spec_data)
            pickle.dump(reducer, open(os.path.join(result_dir, method + '_model.sav'), 'wb'), protocol=4)
        embedding = reducer.transform(spec_data)
    embedding = utils.NormalizeData(embedding)  # normalize to range [0, 1] corresponding to RGB color code
    print('{} finished at {}'.format(method, datetime.now()))

    # clustering
    print('{} started at {}'.format(clustering, datetime.now()))
    if clustering == 'hdbscan':
        labels = HDBSCAN_clustering(data=embedding, min_samples=min_samples, min_cluster_size=min_clusters, start=0)
        # print("labels=", np.unique(labels))
    elif clustering == 'hierarchical':
        labels = hierarchical_clustering(data=embedding, k=n_clusters)
    elif clustering == 'gaussian_mixture':
        labels = gaussian_mixture(data=embedding, k=n_clusters)
    elif clustering == 'k-means':
        labels = kmeans_clustering(data=embedding, k=n_clusters)

    print('{} finished at {}'.format(clustering, datetime.now()))
    # extract RGBA color for labels
    colors = []
    # print(np.unique(labels))

    # transdict = {-1: -1, 0: 0, 1: 0, 2: 1, 3: 0, 4: 2, 5: 2, 6: 3, 7: 2}
    # labels = [transdict[letter] for letter in labels]
    # if len(np.unique(labels)) <= 10:
    #     tab10_colors = [plt.cm.tab10(i) for i in range(len(np.unique(labels)))]
    #     # tab10_colors.reverse()
    #     #tab10_colors = [(1, 0, 0, 1), (1, 0, 1, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    #     #print(tab10_colors)
    #     #tab10_colors = [tab10_colors[0], tab10_colors[2], tab10_colors[1], tab10_colors[3]]
    #     for label in labels:
    #         rgba = tab10_colors[label]
    #         colors.append(rgba)
    # else:
    # tab_colors = list(mcolors.TABLEAU_COLORS.values())
    # print(tab_colors)
    # for label in labels:
    #     if label == -1:
    #         rgba = (0.5, 0.5, 0.5, 0.3)
    #     else:
    #         rgba = mcolors.to_rgba(tab_colors[label])
    #     colors.append(rgba)
    labels_norm = list(utils.NormalizeData(labels))
    cmap = plt.cm.get_cmap(cmap)
    for label, label_norm in zip(labels, labels_norm):
        if label == -1:
            rgba = (0.5, 0.5, 0.5, 0.5)
        else:
            rgba = tuple(cmap(label_norm))
        colors.append(rgba)

    # create data frames with results
    if method == 't-sne':
        col_1 = 'TSNE_1'
        col_2 = 'TSNE_2'
    else:
        col_1 = 'UMAP_1'
        col_2 = 'UMAP_2'

    df_embedding = pd.DataFrame(embedding, columns=[col_1, col_2])
    df_result = pd.concat([df_meta_data, df_embedding], axis=1)
    df_result['label'] = labels
    df_result['label_color'] = colors
    is_labeled = df_result['label'] != -1
    df_labeled_result = df_result[is_labeled]

    # print(np.unique(df_result['label_color'].to_numpy()))

    result_groups_dict = {}
    groups = df_combined['group'].unique()
    for group in groups:
        is_group = df_combined['group'] == group
        df_group_result = df_result[is_group]
        result_groups_dict[group] = df_group_result

    print('saving result to csv...')
    df_result.to_csv(os.path.join(result_dir, method+'_data.csv'), index=False)

    print('saving visualizations...')
    # plot results

    # individual scatterplot for each group
    group_colors = ['tab:orange', 'tab:blue']
    for i, group in enumerate(result_groups_dict.keys()):
        df_group = result_groups_dict.get(group)
        sns.scatterplot(x=col_1, y=col_2, hue='sample', data=df_group, legend='full', linewidth=0, s=dot_size)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('{} data colored by sample no.'.format(group))
        plt.savefig(os.path.join(result_dir, group + '_2D_' + method + '_colored_by_sample.png'), dpi=300)
        plt.close()

        sns.scatterplot(x=col_1, y=col_2, color=group_colors[i], data=df_group, legend='full', linewidth=0, s=dot_size)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('{} data colored by sample no.'.format(group))
        plt.savefig(os.path.join(result_dir, group + '_2D_' + method + '_colored_by_group.png'), dpi=300)
        plt.close()


    # individual scatterplot for each sample
    samples = df_result['sample'].to_numpy()
    samples = np.unique(samples)
    for sample in samples:
        sample_df = df_result[df_result['sample'] == sample]
        # if np.unique(sample_df['group'].to_numpy())[0] == 'Urothelium':
        #     color = 'steelblue'
        # else:
        #     color = 'darkorange'
        # color = ['steelblue', 'darkorange']
        # print(sample_df)
        sns.scatterplot(x=col_1, y=col_2, data=sample_df, legend='full', linewidth=0, hue='group', s=dot_size)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        plt.title('Sample {}'.format(sample))
        plt.savefig(os.path.join(result_dir, method + '_sample_{}.png'.format(sample)), dpi=300)
        plt.close()

    # create hyperspectral visualization according to clusters
    'saving hyperstpectral visualizations....'
    for group in groups:
        group_file_list = [f for f in file_list if group in f]
        plot_clustered_imgs(file_dir=imzml_dir, file_list=group_file_list, df_data=df_result, result_dir=result_dir)
        extract_binary_imgs(file_dir=imzml_dir, file_list=group_file_list, df_data=df_result, result_dir=result_dir)


    # scatterplots of complete data
    sns.scatterplot(x=col_1, y=col_2, hue='sample', data=df_result, legend='full', linewidth=0, s=dot_size)
    plt.title('All data colored by sample no.')
    plt.savefig(os.path.join(result_dir, 'combined_2D_'+method+'_colored_by_sample.png'), dpi=300)
    plt.close()
    sns.scatterplot(x=col_1, y=col_2, hue='group', data=df_result, legend='full', linewidth=0, palette=['darkorange', 'steelblue'], s=dot_size)
    plt.title('All data colored by group')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')
    plt.savefig(os.path.join(result_dir, 'combined_2D_'+method+'_colored_by_group.png'), dpi=300)
    plt.savefig(os.path.join(result_dir, 'combined_2D_' + method + '_colored_by_group.svg'))
    plt.close()
    # fig = px.scatter(df_result, x=col_1, y=col_2, color='group', hover_data=['group', 'sample'])
    # fig.write_html(os.path.join(result_dir, 'combined_2D_'+method+'_colored_by_group.html'))
    # plot = df_result.hvplot.scatter(x=col_1, y=col_2, by='group', legend='right', hover_cols=['group', 'sample'])
    # hvplot.save(plot, os.path.join(result_dir, 'combined_2D_'+method+'_colored_by_group.html'))

    # cluster scatter plot
    # create color_dict for scatter plot
    df_label_info = df_result.drop_duplicates(subset=['label'])
    # print(df_result)
    # print(df_label_info)
    color_dict = {}
    for ind in df_label_info.index:
        color_dict[df_label_info['label'][ind]] = df_label_info['label_color'][ind]
    if len(set(labels)) > 20:
        leg = False
    else:
        leg = 'full'
    # print(color_dict)
    # sns.scatterplot(x=col_1, y=col_2, hue='label', data=df_labeled_result, legend=leg, linewidth=0,
    #                 palette=color_dict, s=dot_size, cmap=cmap)
    # sns.scatterplot(x=col_1, y=col_2, hue='label', data=df_result, legend=leg, linewidth=0,
    #                 palette=color_dict, s=dot_size, cmap=cmap)
    # print(df_result[~is_labeled])
    # print(df_result[is_labeled])
    sns.scatterplot(x=col_1, y=col_2, data=df_result[~is_labeled], legend=leg, linewidth=0,
                    s=dot_size, c=df_result[~is_labeled]['label_color'])
    sns.scatterplot(x=col_1, y=col_2, data=df_result[is_labeled], legend=leg, linewidth=0,
                    s=dot_size, c=df_result[is_labeled]['label_color'])

    plt.title('All data colored class label')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')
    plt.savefig(os.path.join(result_dir, 'clusters.png'), dpi=300)
    plt.savefig(os.path.join(result_dir, 'clusters.svg'))
    plt.close()

    # individual scatterplot for each group
    for group in result_groups_dict.keys():
        df_group = result_groups_dict.get(group)
        # print(df_group)
        sns.scatterplot(x=col_1, y=col_2, data=df_group[~is_labeled], legend=leg, linewidth=0,
                        s=dot_size, c=df_group[~is_labeled]['label_color'])
        sns.scatterplot(x=col_1, y=col_2, data=df_group[is_labeled], legend=leg, linewidth=0,
                        s=dot_size, c=df_group[is_labeled]['label_color'])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('{} data colored by sample no.'.format(group))
        plt.savefig(os.path.join(result_dir, group + '_2D_' + method + '_clusters.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs dimensionality reduction and clustering on multiple mass '
                                                 'spectrometry imaging data sets')
    parser.add_argument('imzML_dir', type=str, help='directory containing imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store results')
    parser.add_argument('-method', type=str, default='umap', help='method for dimensionality reduction')
    parser.add_argument('-dist_metric', type=str, default='cosine', help='distance metric for umap')
    parser.add_argument('-n_neighbors', type=int, default=50, help='number of neighbors for umap')
    parser.add_argument('-min_dist', type=float, default=0.0, help='min_dist for UMAP')
    parser.add_argument('-clustering_method', type=str, default='hdbscan', help='method for clustering')
    parser.add_argument('-min_cluster_size', type=int, default=1000, help='min number of clusters for hdbscan')
    parser.add_argument('-min_samples', type=int, default=5, help='min number of samples for hdbscan')
    parser.add_argument('-n_clusters', type=int, default=10, help='number of clusters for k-means')
    parser.add_argument('-preembedding', type=bool, default=False, help='set to True to perform preembedding with TSVD')
    parser.add_argument('-preembedding_model', type=str, default='', help='model of preembedding')
    parser.add_argument('-embedding_model', type=str, default='', help='model of embedding')
    parser.add_argument('-dot_size', type=float, default=0.5, help='sot size for scatter plots')
    parser.add_argument('-cmap', type=str, default='Spectral', help='cmap')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(args.imzML_dir, args.method + '_' + str(args.n_neighbors) + 'neighb_')
        if args.clustering_method != 'hdbscan':
            args.result_dir += args.clustering_method + '_' + str(args.n_clusters) + 'clusters'
        elif args.clustering_method == 'hdbscan':
            args.result_dir += 'hdbscan_' + str(args.min_cluster_size) + 'mcs_' + str(args.min_samples) + 'ms'
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    print('saving results to ', args.result_dir)

    umap_groups(imzml_dir=args.imzML_dir, result_dir=args.result_dir, method=args.method, dist_metric=args.dist_metric,
                n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                clustering=args.clustering_method, min_clusters=args.min_cluster_size, min_samples=args.min_samples,
                n_clusters=args.n_clusters, preembedding=args.preembedding, preembedding_model=args.preembedding_model,
                embedding_model=args.embedding_model, dot_size=args.dot_size, cmap=args.cmap)
