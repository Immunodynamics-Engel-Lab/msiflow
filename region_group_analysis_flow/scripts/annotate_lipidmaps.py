import argparse
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate a csv file containing m/z values and sample intensities with lipidmaps annotations')
    parser.add_argument('mz_file', type=str, help='file with measured m/z values')
    parser.add_argument('lipidmaps_file', type=str, help='lipidmaps file containing potential lipid matches')
    #parser.add_argument('merge_col', type=str, help='column name for merging')
    parser.add_argument('-cols_sec_file', help='columns to keep from lipidmaps file', default='Input Mass,Name',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('output_file', type=str, help='output file path')
    args = parser.parse_args()

    df1 = pd.read_csv(args.mz_file)
    df2 = pd.read_csv(args.lipidmaps_file, delimiter='\t')
    df1[df1.columns[0]] = df1[df1.columns[0]].astype(np.float32)
    df2[df2.columns[0]] = df2[df2.columns[0]].astype(np.float32)

    #cols_df2 = [df2.columns[0]]
    #cols_df2.extend(args.cols_sec_file)
    cols_df2 = args.cols_sec_file
    df2 = df2[cols_df2]
    df2.dropna(inplace=True)

    merged = pd.merge(df1, df2, left_on=[df1.columns[0]], right_on=df2.columns[0], how='inner')
    merged.drop(columns=df2.columns[0], inplace=True)

    merged2 = merged.groupby(merged.columns[0])['Name'].apply(lambda x: ','.join(x)).reset_index()
    #print(merged2)
    #merged = merged.groupby(merged.columns[0])['Name'].apply(lambda x: ','.join(x)).reset_index()
    #print(merged)
    # final_merged = pd.merge(df1, merged, left_on=df1.columns[0], right_on=merged.columns[0], how='inner')
    # print(final_merged)
    #final_merged = pd.merge(merged, merged2, left_on=df1.columns[0], right_on=merged.columns[0], how='inner')
    merged.drop(columns=['Name'], inplace=True)
    final_merged = pd.merge(merged, merged2, left_on=merged.columns[0], right_on=merged2.columns[0], how='inner')
    final_merged.drop_duplicates(ignore_index=True, inplace=True)

    final_merged['Lipid class'] = final_merged['Name'].str.split(" ").str[0]
    #print(final_merged)

    lipid_classes = np.unique(final_merged['Lipid class'].to_numpy())
    class_no = np.arange(0, lipid_classes.shape[0])
    # print(lipid_classes)
    # print(class_no)
    lipid_class_no_dict = dict(zip(lipid_classes, class_no))
    final_merged['Lipid class no.'] = final_merged['Lipid class'].map(lipid_class_no_dict)
    # print(final_merged)
    # final_merged['Lipid class no.'] = 0
    #
    # final_merged['Lipid class no.'][final_merged['Lipid class'] == 'DG'] = 1
    # final_merged['Lipid class no.'][final_merged['Lipid class'] == 'PC'] = 2
    # final_merged['Lipid class no.'][final_merged['Lipid class'] == 'PE'] = 3
    # final_merged['Lipid class no.'][final_merged['Lipid class'] == 'SFE'] = 4
    # final_merged['Lipid class no.'][final_merged['Lipid class'] == 'SM'] = 5
    # final_merged['Lipid class no.'][final_merged['Lipid class'] == 'ST'] = 6
    # final_merged['Lipid class no.'][final_merged['Lipid class'] == 'TG'] = 6
    # print(final_merged)
    # # df1 = pd.read_csv(args.file1, index_col=0)
    # # df2 = pd.read_csv(args.file2, index_col=0)
    # # df2.rename(columns={df2.columns[-1]: 'correlation'}, inplace=True)
    # # df2 = df2['correlation']
    # #
    # # print(df1)
    # # print(df2)
    # #
    # # merged = df1.join(df2)
    # # print(merged)
    # #
    # print(final_merged)
    final_merged.to_csv(args.output_file, index=False)
