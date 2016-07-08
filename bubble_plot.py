import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import os

#
SAMPLE_ANNOTATION_FILE = 'sample_annotation_file'
COUNT_FILE = 'count_file'
DISPLAY_COUNT = 'display_count'
DGE_FILE = 'dge_file'
SEQ_DEPTH_FILE = 'seq_depth_file'
FDR_THRESHOLD = 'fdr_threshold'
GROUP_1 = 'g1'
GROUP_2 = 'g2'

DEFAULT_FDR = 0.05

DEFAULT_COLORS = sns.xkcd_palette(["windows blue",
                                   "amber",
                                   "greyish",
                                   "faded green",
                                   "dusty purple",
                                   "pale blue",
                                   "green yellow",
                                   "pumpkin"])

cc = mpl.colors.ColorConverter()
DEFAULT_COLORS = cc.to_rgba_array(DEFAULT_COLORS, alpha=0.5)

class MakeAbsolutePathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.realpath(os.path.abspath(values)))


def parse_cl_args():
    '''
    Parses the command line args
    :return: dictionary of commandline options
    '''
    parser = argparse.ArgumentParser()

    # file with two columns (Tab-delim) with sample name mapping to experimental group
    parser.add_argument("-s",
                        "--samples",
                        required = True,
                        action = MakeAbsolutePathAction,
                        dest = SAMPLE_ANNOTATION_FILE,
                        help='Two-column, tab delimited file with samples in first column and group identifiers in'
                             'second column.  Needs columns to be labeled as "sample" and "group"')

    # count matrix- column names give the sample names.  Gene IDs index the rows
    parser.add_argument("-c",
                        "--counts",
                        required = True,
                        action = MakeAbsolutePathAction,
                        dest = COUNT_FILE,
                        help='The count matrix, tab-delimited.  Column headers should be the sample names, except the '
                             'first column, which has the gene names.  That column can be named anything or follow '
                             'the R convention of having an initial "tab" space.')

    # how many genes to show
    parser.add_argument("-n",
                        "--number",
                        required = True,
                        type = int,
                        dest = DISPLAY_COUNT,
                        help='An integer- how many plots to make.')

    # a file with the differential expression results (DESeq format)
    parser.add_argument("-d",
                        "--dge_file",
                        required = True,
                        action = MakeAbsolutePathAction,
                        dest = DGE_FILE,
                        help='The file of differential expression results, in "DESeq" format.  We required that'
                             ' it be comma-delimited, have a "log2FoldChange" column, and a "padj" column')

    # a two-column (tab-delim) file with the sample name mapped to the total mapped reads (or other proxy for seq depth)
    # not required- if it's there, then the bubbles are sized according to the sequencing depth
    parser.add_argument("-sd",
                        "--seq_depths",
                        required = False,
                        action = MakeAbsolutePathAction,
                        dest = SEQ_DEPTH_FILE,
                        help='Two-column, tab delimited file with samples in first column and read count in '
                             'second column.  Needs columns to be labeled as "sample" and "reads".  Used'
                             ' to size the points in the bubble plot.  If not supplied, all points will be'
                             ' the same size.')

    # will only plot dge genes with FDR less than this.
    parser.add_argument("-q",
                        "--fdr",
                        required = False,
                        type = float,
                        default = DEFAULT_FDR,
                        dest = FDR_THRESHOLD,
                        help='The adjusted p-value (FDR) threshold.  If the number of genes is less '
                             'than the "n" argument above, then this supersedes it.')

    parser.add_argument("-g1",
                        "--group1",
                        required = True,
                        dest = GROUP_1,
                        help='String identifying a group.  Needs to match one of the groups from the sample'
                             ' annotation file.')

    parser.add_argument("-g2",
                        "--group2",
                        required = True,
                        dest = GROUP_2,
                        help='String identifying a group.  Needs to match one of the groups from the sample'
                             ' annotation file.')

    return vars(parser.parse_args())


def set_plot_style():
    """
    Sets up some plot styles.  Add/subtract anything as necessary
    """
    sns.set_style('darkgrid',rc={'font.family':'serif'})
    font = {'family':'serif', 'size':24}
    mpl.rc('font', **font)


def make_boxplot(ax, data_dict, pointsize_fractions, gene=None, lfc=None, p=None, randomize=False):

    plot_data = []
    labels = []

    # the sizes of the circles:
    minsize = 40
    maxsize=400

    # in addition to making the points different sizes based on seq depth,
    # also place the points horizontally based on this depth (rather than random scatter).
    #dx is width of this 'band'.  If dx=0.5, then horizontal values would lie in (0.75,1.25)
    dx = 0.5

    point_sizes = minsize + pointsize_fractions*(maxsize-minsize)
    if randomize:
        pointsize_fractions = pd.Series(np.random.random(size=pointsize_fractions.shape[0]),index = pointsize_fractions.index.values)
    relative_point_locs = -0.5*dx + pointsize_fractions*dx

    for key, vals in data_dict.items():
        plot_data.append(vals)
        labels.append(key)

    for i in range(len(labels)):

        group_data = data_dict[labels[i]]
        s = point_sizes.ix[group_data.index.values]
        point_locs = i - relative_point_locs.ix[group_data.index.values]
        ax.scatter(point_locs,group_data,s=s, c=DEFAULT_COLORS[i+2])

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim((-1.1*dx, 1+1.1*dx))
    ax.set_ylabel('Normalized Expression')
    ax.set_title('%s\n$\log_2$FC=%.3f\nq=%.2E' % (gene, lfc, p))


def make_plot(dge_results, count_mtx, group_to_sample_dict, group1, group2, seq_depth_df):
    n = dge_results.shape[0]
    ncols = 3
    if (n % ncols) == 0:
        nrows = n/ncols
    else:
        nrows = n/ncols+1
    fig, axarray = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,n*1.6))

    all_samples = group_to_sample_dict[group1] + group_to_sample_dict[group2]
    if seq_depth_df is None:
        r = pd.Series(np.repeat(1.0, len(all_samples)), index=all_samples, name='reads')
        randomize = True
    else:
        # now get the min/max of those seq depths
        t0 = seq_depth_df.reads.min()
        t1 = seq_depth_df.reads.max()
        r = (seq_depth_df.reads - t0)/float(t1-t0)
        randomize = False

    for i,(idx, row) in enumerate(dge_results.iterrows()):
        row_num = i/ncols
        col_num = i%ncols
        counts = count_mtx.ix[idx]

        d = {}
        d[group1] = counts[group_to_sample_dict[group1]]
        d[group2] = counts[group_to_sample_dict[group2]]

        ax = axarray[row_num, col_num]
        make_boxplot(ax, d, r, gene=idx, lfc=row['log2FoldChange'], p=row['padj'], randomize = randomize)
    plt.tight_layout()
    fig.savefig('top_dge_hits.pdf')
    plt.close()


def get_dge_data(cl_args_dict):
    """
    Loads and preps/filters the differential expression result dataframe
    """
    dge_results = pd.read_table(cl_args_dict.get(DGE_FILE), sep=',', index_col=0)
    dge_results.sort('padj', ascending=True, inplace=True)

    # filter for diff exp genes (by supplied or default threshold)
    dge_results = dge_results.ix[dge_results.padj <= cl_args_dict.get(FDR_THRESHOLD)]

    # if we still have more genes than we want to plot, take the top N
    if dge_results.shape[0] > cl_args_dict.get(DISPLAY_COUNT):
        dge_results = dge_results.iloc[:cl_args_dict.get(DISPLAY_COUNT)]

    return dge_results


def get_counts(cl_args_dict):
    """
    Loads and preps the counts
    """
    count_mtx = pd.read_table(cl_args_dict.get(COUNT_FILE), index_col = 0)
    return count_mtx


def get_library_sizes(cl_args_dict):
    if cl_args_dict.get(SEQ_DEPTH_FILE):
        return pd.read_table(cl_args_dict.get(SEQ_DEPTH_FILE), index_col = 0)
    else:
        return None


def get_sample_annotations(cl_args_dict):
    d = pd.read_table(cl_args_dict.get(SAMPLE_ANNOTATION_FILE))
    groupings = d.groupby('group')
    group_to_samples_dict = {}
    for g in groupings:
        group_to_samples_dict[g[0]] = g[1].sample.values.tolist()
    return group_to_samples_dict


def main(cl_args_dict):
    """
    The main method.  Receives a dictionary with the passed commandline args
    """

    group_to_sample_dict = get_sample_annotations(cl_args_dict)
    dge_results = get_dge_data(cl_args_dict)
    count_mtx = get_counts(cl_args_dict)
    library_size_df = get_library_sizes(cl_args_dict)
    make_plot(dge_results, count_mtx, group_to_sample_dict, cl_args_dict.get(GROUP_1), cl_args_dict.get(GROUP_2), library_size_df)





if __name__ == '__main__':
	cl_args = parse_cl_args()
	set_plot_style()
	main(cl_args)




























