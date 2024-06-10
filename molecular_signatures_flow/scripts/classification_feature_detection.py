import tifffile
import numpy as np
import argparse
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from itertools import cycle
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import shap
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

sys.path.append("..")
from pkg import utils


def plot_feat_imp(n, df, output):
    top_feat_imp_df = df.iloc[:n, :]
    top_feat_imp_df['m/z'] = np.round(top_feat_imp_df['m/z'].to_numpy().astype(np.float64), 2)
    sns.barplot(x=top_feat_imp_df['feature importance'], y=top_feat_imp_df['m/z'], orient='h',
                order=top_feat_imp_df['m/z'])
    plt.savefig(output)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find lead masses for a binary image')
    parser.add_argument('bin_img_dir', type=str, help='directory with binary images')
    parser.add_argument('imzML_dir', type=str, help='directory with imzML files of samples')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store results')
    parser.add_argument('-model', type=str, default='XGBoost', help='model',
                        choices=['XGBoost', 'LGBoost', 'AdaBoost', 'CatBoost', 'GBoost', 'RandomForest'])
    parser.add_argument('-model_file', type=str, default='', help='file of saved model')
    parser.add_argument('-balancing_method', type=str, default='standard', help='method to tackle class imbalance',
                        choices=['standard', 'smote', 'undersample', 'oversample', 'weights'])
    parser.add_argument('-num', type=int, default=10, help='number of top hit to plot')
    parser.add_argument('-n_folds', type=int, default=0, help='number of folds for cross validation')
    parser.add_argument('-pos_class', type=str, default='pos', help='if binary classification, set positive class label')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    if args.result_dir == '':
        if args.balancing_method != '':
            dir_name = args.model + '_' + args.balancing_method
        else:
            dir_name = args.model
        args.result_dir = os.path.join(args.bin_img_dir, dir_name)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # read data
    # expected annotation in file names: class_group_sample.tif
    imzML_files = [file for file in os.listdir(args.imzML_dir) if file.endswith('.imzML')]
    imzML_files = np.array(sorted(imzML_files, key=lambda x: int(x.split('.')[0][-2:])))
    file_names = [f.split('.')[0] for f in imzML_files]
    samples = [f.split('.')[0][-2:] for f in imzML_files]
    bin_imgs = [file for file in os.listdir(args.bin_img_dir) if file.endswith('.tif')]
    bin_imgs = np.array(sorted(bin_imgs, key=lambda x: int(x.split('.')[0][-2:])))
    classes = set([f.split('_')[0] for f in bin_imgs])
    #classes = set([f.split('.')[0].split('_')[0] + '_' + f.split('.')[0].split('_')[1] for f in bin_imgs])
    class_labels_dict = {k: i for k, i in enumerate(classes, 0)}

    print('------------------------------')
    print('imzML_files=', imzML_files)
    print('file_names=', file_names)
    print('bin_imgs=', bin_imgs)
    print('samples=', samples)
    print("classes=", classes)
    print("class_labels_dict=", class_labels_dict)

    df = utils.get_combined_dataframe_from_files(args.imzML_dir, imzML_files, multi_index=True)
    #df.columns = np.round(df.columns.to_numpy(), 4)
    df['label'] = [1000] * df.shape[0]
    print(df)

    # add labels to dataframe
    print('adding class labels to dataframe...')
    for fl, smpl in zip(file_names, samples):
        for cl in class_labels_dict.values():
            cl_num = [k for k, v in class_labels_dict.items() if v == cl][0]
            smpl_cl_fl = os.path.join(args.bin_img_dir, cl + '_' + fl + '.tif')
            #smpl_cl_fl = os.path.join(args.bin_img_dir, cl + '_' + smpl + '.tif')
            if os.path.exists(smpl_cl_fl):
                smpl_cl_img = tifffile.imread(os.path.join(args.bin_img_dir, smpl_cl_fl))
                smpl_cl_px_idx_np = np.nonzero(smpl_cl_img)
                num_smpl_cl_px = np.count_nonzero(smpl_cl_img)
                smpl_cl_px_idx = list(tuple(zip([fl] * num_smpl_cl_px, smpl_cl_px_idx_np[1], smpl_cl_px_idx_np[0])))
                #df.loc[smpl_cl_px_idx, 'label'] = cl
                #intersecting_idx = [key for key in smpl_cl_px_idx if key in df.index.to_list()] # in case pixel no MSI pixel
                #df.loc[intersecting_idx, 'label'] = cl_num
                df.loc[smpl_cl_px_idx, 'label'] = cl_num
    df = df[df['label'] != 1000]

    #y = df['label'].to_numpy()
    y = df['label']
    print("labels=", np.unique(y.to_numpy()))
    _, idx = np.unique(y.to_numpy(), return_index=True)
    class_names = y.to_numpy()[np.sort(idx)]
    print("class_names=", class_names)
    #X = df.iloc[:, :-1].to_numpy()
    X = df.iloc[:, :-1]

    print("y=", y)
    print("X=", X)

    # select model
    if args.model == 'RandomForest':
        model = RandomForestClassifier()
    elif args.model == 'AdaBoost':
        model = AdaBoostClassifier()
    elif args.model == 'GBoost':
        model = GradientBoostingClassifier()
    elif args.model == 'LGBoost':
        data = lgb.Dataset(X, label=y)
        params = {'is_unbalance': 'true',
                  'boosting': 'gbdt',
                  'objective': 'multiclass',
                  'metric': 'multi_logloss',
                  'max_depth': 10,
                  'num_class': len(classes)
                  }
    elif args.model == 'CatBoost':
        model = ctb.CatBoostClassifier()
    else:
        #from sklearn.preprocessing import LabelEncoder
        #le = LabelEncoder()
        #y = le.fit_transform(y)
        # model = xgb.XGBClassifier(device='cuda')
        model = xgb.XGBClassifier(n_jobs=-1)

    # model = RandomForestClassifier()
    #classifier = cross_validate(model, X, y, cv=cv, scoring='accuracy', return_estimator=True, n_jobs=-1)
    #print(classifier)

    # apply class imbalance method
    sample_weights = None
    if args.balancing_method != 'standard' and args.balancing_method != 'weights':
        if args.balancing_method == 'smote':
            sampler = SMOTE(random_state=42)
        if args.balancing_method == 'undersample':
            sampler = RandomUnderSampler(sampling_strategy='majority')
        if args.balancing_method == 'oversample':
            sampler = RandomOverSampler(sampling_strategy='minority')


    # train one classifier and retrieve feature importances
    if args.n_folds == 0:
        print('------------------------------')
        print('performing classification...')

        if args.balancing_method != 'standard':
            if args.balancing_method != 'weights':
                X, y = sampler.fit_resample(X, y)
            else:
                sample_weights = compute_sample_weight(class_weight='balanced', y=y)

        if args.model_file == '':
            if args.model == 'LGBoost':
                classifier = lgb.train(params, data)
            else:
                classifier = model.fit(X, y, sample_weight=sample_weights)
            pickle.dump(classifier, open(os.path.join(args.result_dir, args.model + '.json'), 'wb'))
        else:
            classifier = pickle.load(open(args.model_file, 'rb'))
        print(classifier)

        # extract mean feature importances
        print('------------------------------')
        print('extracting feature importances...')
        if args.model == 'LGBoost':
            mean_feature_importance = classifier.feature_importance()
        else:
            mean_feature_importance = classifier.feature_importances_
        mean_feature_importance = MinMaxScaler(feature_range=(0, 1)).fit_transform(mean_feature_importance.reshape(-1, 1))
        # print(mean_feature_importance)
        # print(classifier)

        feat_imp_df = pd.DataFrame.from_dict(
            {'m/z': df.columns[:-1], 'feature importance': mean_feature_importance.flatten()})
        feat_imp_df.sort_values(by=['feature importance'], ascending=False, inplace=True, ignore_index=True)
        feat_imp_df.to_csv(os.path.join(args.result_dir, args.model + '_feature_importance.csv'), index=False)

        # plot average feature importance
        plot_feat_imp(n=args.num,
                      df=feat_imp_df,
                      output=os.path.join(args.result_dir, args.model + '_feature_importance.svg')
                      )

        # compute SHAP values
        print('------------------------------')
        print('extracting SHAP values...')
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X)

        # summary plot based on shap values
        shap.summary_plot(shap_values, X.values, plot_type="bar",
                          class_names=[class_labels_dict[i] for i in model.classes_],
                          feature_names=X.columns, show=False)
        plt.savefig(os.path.join(args.result_dir, args.model + '_shap_summary.svg'))
        plt.close()

        if len(class_names) > 2:
            # summary plot based on feature importance
            features_list = feat_imp_df['m/z'].head(args.num)
            feature_col_ids = [df.columns.get_loc(c) for c in features_list if c in df.columns]

            shap_values_fl = []
            for i in range(len(shap_values)):
                shap_vals = shap_values[i]
                shap_vals_fl = np.take(shap_vals, feature_col_ids, axis=1)
                shap_values_fl.append(shap_vals_fl)

            X_fl = X[features_list]
            shap.summary_plot(shap_values_fl, X_fl, plot_type='bar',
                              class_names=[class_labels_dict[i] for i in model.classes_], feature_names=X_fl.columns,
                              show=False)
            plt.savefig(os.path.join(args.result_dir, args.model + '_shap_summary_feat_imp.svg'))
            plt.close()

            # shap plot for each class
            for i, cl in enumerate(model.classes_):
                print(class_labels_dict[cl])
                shap.summary_plot(shap_values[i], X.values, feature_names=X.columns, show=False)
                plt.savefig(
                    os.path.join(args.result_dir, args.model + '_shap_summary_' + class_labels_dict[cl] + '.svg'))
                plt.close()
        else:
            shap.summary_plot(shap_values, X.values, feature_names=X.columns, show=False)
            plt.savefig(os.path.join(args.result_dir, args.model + '_shap_summary_violin.svg'))
            plt.close()
    # cross-validation
    else:
        n_splits = args.n_folds
        skf = StratifiedKFold(n_splits=n_splits)

        if len(class_names == 2):
            pos_cl_num = [k for k, v in class_labels_dict.items() if v == args.pos_class][0]
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            fig, ax = plt.subplots(figsize=(6, 6))
            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                train_x, train_y, test_x, test_y = (X.iloc[train_index, :], y.iloc[train_index],
                                                    X.iloc[test_index, :], y.iloc[test_index])
                if args.balancing_method != 'standard' and args.balancing_method != 'weights':
                    resampl_train_x, resampl_train_y = sampler.fit_resample(train_x, train_y)
                    model.fit(resampl_train_x, resampl_train_y, sample_weight=sample_weights)
                else:
                    if args.balancing_method == 'weights':
                        sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)
                    model.fit(train_x, train_y, sample_weight=sample_weights)
                viz = RocCurveDisplay.from_estimator(model, test_x, test_y, name=f"ROC fold {fold}", alpha=0.3, lw=1,
                                                     ax=ax, pos_label=pos_cl_num)
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                    lw=2, alpha=0.8, )
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.", )
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel="False Positive Rate", ylabel="True Positive Rate",
                   title=f"Mean ROC curve with variability\n(Positive label '{args.pos_class}')", )
            ax.axis("square")
            ax.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
        else:
            y_bin = label_binarize(y, classes=class_names)
            n_classes = y_bin.shape[1]
            class_labels = [class_labels_dict[i] for i in class_names]
            y_score = cross_val_predict(model, X, y, cv=skf, method='predict_proba', n_jobs=-1)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            colors = cycle([plt.cm.tab10(i) for i in range(class_names.shape[0])])
            # colors = cycle(['blue', 'red', 'green'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(class_labels[i], roc_auc[i]))

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

            # Compute macro-average ROC curve and ROC area
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            # Interpolate all ROC curves at these points
            mean_tpr = np.zeros_like(fpr_grid)
            for i in range(n_classes):
                mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
            # Average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = fpr_grid
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.result_dir, args.model + '_' + args.balancing_method + '_ROC_curve.svg'))

        if args.plot:
            plt.show()






    # for idx, estimator in enumerate(classifier['estimator']):
    #     print(estimator)
    #     print([i for i in estimator.__dict__ if i.endswith('_')])
    # _, idx = np.unique(y, return_index=True)
    # print(y[np.sort(idx)])
    # y_bin = label_binarize(y, classes=[1, 2, 3])
    # n_classes = y_bin.shape[1]
    #



    # # create directory for top m/z and save image for each imzml file
    # print('saving top m/z images...')
    # for i, mz in enumerate(tqdm(top_mzs)):
    #     mz_dir = os.path.join(args.result_dir, str(round(mz, 4)).replace('.', '_') + 'mz')
    #     if not os.path.exists(mz_dir):
    #         os.mkdir(mz_dir)
    #     for imzML_fl in imzML_files:
    #         sample_num = imzML_fl.split('.')[0][-2:]
    #         p = ImzMLParser(os.path.join(args.imzML_dir, imzML_fl))
    #         pyx = (p.imzmldict["max count of pixels y"] + 1, p.imzmldict["max count of pixels x"] + 1)
    #         msi_df = utils.get_dataframe_from_imzML(os.path.join(args.imzML_dir, imzML_fl), multi_index=True)
    #         top_mz_img = get_mz_img(pyx, msi_df, mz)
    #         if args.contrast_stretch:
    #             p0, p99 = np.percentile(top_mz_img, (0, 99.9))
    #             top_mz_img = rescale_intensity(top_mz_img, in_range=(p0, p99))
    #         top_mz_img = (utils.NormalizeData(top_mz_img) * 255).astype('uint8')
    #         tifffile.imwrite(os.path.join(mz_dir, sample_num + '.tif'), data=top_mz_img)



    # # get target values
    # bin_imgs = [file for file in os.listdir(args.bin_img_dir) if file.endswith('.tif')]
    # bin_imgs = np.array(sorted(bin_imgs, key=lambda x: int(x.split('.')[0][-2:])))
    # print(bin_imgs)
    # y = []
    # for fl in bin_imgs:
    #     bin_img = NormalizeData(tifffile.imread(os.path.join(args.bin_img_dir, fl)))
    #     bin_img = bin_img.ravel()
    #     y.extend(bin_img.tolist())
    # y = np.asarray(y)
    #
    # # get features (ion images) for all samples
    # imzML_files = [file for file in os.listdir(args.imzML_dir) if file.endswith('.imzML')]
    # imzML_files = np.array(sorted(imzML_files, key=lambda x: int(x.split('.')[0][-2:])))
    # samples = [fl.split('.')[0][-2:] for fl in imzML_files]
    # print(samples)
    #
    # print("generating X with all ion images")
    # all_mz_imgs = []
    # for i, smpl in enumerate(samples):
    #     sample_mz_imgs = []
    #     p = ImzMLParser(os.path.join(args.imzML_dir, imzML_files[i]))
    #     pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    #     df = get_dataframe_from_imzML(os.path.join(args.imzML_dir, imzML_files[i]), multi_index=True)
    #     coords = df.index.tolist()
    #     for mz in df.columns:
    #         mz_img = np.zeros(pyx).astype(np.uint8)
    #         ints = df[mz].to_numpy()
    #         for j, (x_val, y_val) in enumerate(coords):
    #             mz_img[y_val, x_val] = ints[j]
    #         mz_img = NormalizeData(mz_img.ravel())
    #         sample_mz_imgs.append(mz_img)
    #     sample_mz_imgs = np.transpose(np.asarray(sample_mz_imgs))
    #     all_mz_imgs.append(sample_mz_imgs)
    # X = np.vstack(all_mz_imgs)
    # print(X.shape)
    # print(y.shape)
    #
    # cv=StratifiedKFold(5)
    # model = AdaBoostClassifier()
    #
    # if args.feature_selection:
    #     # recursive feature elimination
    #     rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy', min_features_to_select=1, n_jobs=-1)
    #     rfecv.fit(X, y)
    #
    #     features = df.columns.to_numpy()
    #     best_features = features[rfecv.support_]
    #     print(f"Optimal number of features: {rfecv.n_features_}")
    #     print("Selected best: ", best_features)
    #
    #     best_features_df = pd.DataFrame.from_dict({'m/z': best_features})
    #     best_features_df.to_csv(os.path.join(args.result_dir, 'best_features.csv'))
    #
    #     min_features_to_select = 1
    #     n_scores = len(rfecv.cv_results_["mean_test_score"])
    #
    #     plt.figure()
    #     plt.xlabel("Number of features selected")
    #     plt.ylabel("Mean test accuracy")
    #     plt.errorbar(
    #         range(min_features_to_select, n_scores + min_features_to_select),
    #         rfecv.cv_results_["mean_test_score"],
    #         yerr=rfecv.cv_results_["std_test_score"],
    #     )
    #     plt.title("Recursive Feature Elimination \nwith correlated features")
    #     plt.savefig(os.path.join(args.result_dir, 'recursive_feature_elimiation.svg'))
    #
    #     top_mzs = best_features
    # elif args.boruta:
    #     boruta = BorutaPy(estimator=model, n_estimators='auto', max_iter=50, verbose=True)
    #     boruta.fit(X, y)
    #
    #     mzs = df.columns.to_numpy()
    #     print(np.unique(boruta.support_))
    #     res = mzs[boruta.support_ > 0]
    #     print("Best m/z values: {}".format(res))
    #     #print("Ranking: ", boruta.ranking_)
    #     df_ranking = pd.DataFrame.from_dict({'m/z': mzs, 'ranking': boruta.ranking_})
    #     df_ranking = df_ranking.sort_values(by=['ranking'])
    #     df_ranking.to_csv(os.path.join(args.result_dir, 'mz_ranking.csv'))
    #
    #     if not res:
    #         res = mzs[boruta.support_weak_ > 0]
    #         print("Tentative m/z values: {}".format(res))
    #
    #     df_res = pd.DataFrame.from_dict({'m/z': res})
    #     df_res.to_csv(os.path.join(args.result_dir, 'selected_mz.csv'))
    # else:
    #     output = cross_validate(model, X, y, cv=cv, scoring='jaccard', return_estimator=True, n_jobs=-1)
    #     feature_importance_list = []
    #     for idx, estimator in enumerate(output['estimator']):
    #         feature_importances = estimator.feature_importances_
    #         feature_importance_list.append(feature_importances)
    #     feature_importance = np.asarray(feature_importance_list)
    #     print(feature_importance)
    #     mean_feature_importance = np.mean(feature_importance, axis=0)
    #     print(mean_feature_importance)
    #     mean_feature_importance = MinMaxScaler(feature_range=(0, 1)).fit_transform(mean_feature_importance.reshape(-1, 1))
    #
    #     feat_imp_df = pd.DataFrame.from_dict({'m/z': df.columns, 'feature importance': mean_feature_importance.flatten()})
    #     feat_imp_df.sort_values(by=['feature importance'], ascending=False, inplace=True, ignore_index=True)
    #
    #     feat_imp_df.to_csv(os.path.join(args.result_dir, 'AdaBoost_feature_importance.csv'))
    #
    #     top_feat_imp_df = feat_imp_df.iloc[:20, :]
    #     top_mzs = top_feat_imp_df['m/z'].to_numpy()
    #     print(top_feat_imp_df)
    #     top_feat_imp_df['m/z'] = np.round(top_feat_imp_df['m/z'].to_numpy(), 2)
    #     sns.barplot(x=top_feat_imp_df['feature importance'], y=top_feat_imp_df['m/z'], orient='h',
    #                 order=top_feat_imp_df['m/z'])
    #     plt.savefig(os.path.join(args.result_dir, 'AdaBoost_feature_importance.svg'))
    #
    # # create directory for top m/z and save image for each imzml file
    # print('saving top m/z images...')
    # for i, mz in enumerate(tqdm(top_mzs)):
    #     mz_dir = os.path.join(args.result_dir, str(round(mz, 4)).replace('.', '_') + 'mz')
    #     if not os.path.exists(mz_dir):
    #         os.mkdir(mz_dir)
    #     for imzML_fl in imzML_files:
    #         sample_num = imzML_fl.split('.')[0][-2:]
    #         p = ImzMLParser(os.path.join(args.imzML_dir, imzML_fl))
    #         pyx = (p.imzmldict["max count of pixels y"] + 1, p.imzmldict["max count of pixels x"] + 1)
    #         msi_df = utils.get_dataframe_from_imzML(os.path.join(args.imzML_dir, imzML_fl), multi_index=True)
    #         top_mz_img = get_mz_img(pyx, msi_df, mz)
    #         if args.contrast_stretch:
    #             p0, p99 = np.percentile(top_mz_img, (0, 99.9))
    #             top_mz_img = rescale_intensity(top_mz_img, in_range=(p0, p99))
    #         top_mz_img = (utils.NormalizeData(top_mz_img) * 255).astype('uint8')
    #         tifffile.imwrite(os.path.join(mz_dir, sample_num + '.tif'), data=top_mz_img)




