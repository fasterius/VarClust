#!/usr/bin/env python3

# Import modules
import argparse
import pandas as pd
import cluster_profiles

# Argument parser -------------------------------------------------------------

parser = argparse.ArgumentParser(epilog='Hierarchically clusters and plots a '
                                        'distance matrix using the chosen '
                                        'method.')
parser.add_argument('input',
                    type=str,
                    help='input distance matrix file path')
parser.add_argument('output',
                    type=str,
                    help='output image file path')
parser.add_argument('-p', '--patients-only',
                    action='store_true',
                    dest='patients_only',
                    help='only cluster on patients, regardless of cell type')
parser.add_argument('-r', '--remove-na-threshold',
                    type=int,
                    default=5,
                    dest='remove_na_threshold',
                    metavar='',
                    help='remove rows/columns containing NAs >= threshold')
parser.add_argument('-m', '--metadata-file',
                    type=str,
                    dest='metadata_file',
                    default=None,
                    metavar='',
                    help='metadata file path')
parser.add_argument('-i', '--metadata-id',
                    type=str,
                    dest='metadata_id',
                    default='SRR',
                    metavar='',
                    help='metadata id column to merge on [default: SRR]')
parser.add_argument('-l', '--linkage-method',
                    type=str,
                    dest='linkage_method',
                    default='complete',
                    metavar='',
                    help='[complete (default), single, average, centroid, '
                         'median, ward, weighted]')
parser.add_argument('-f', '--filter',
                    type=str,
                    dest='filter',
                    default='none',
                    metavar='',
                    help='filter cells [none (default), low_quality, '
                         'low_quality_and_bulk]')
parser.add_argument('-c', '--colour-threshold',
                    type=int,
                    dest='colour_threshold',
                    default=0,
                    metavar='',
                    help='threshold for branch colours [default: 0]')
parser.add_argument('-g', '--group-cols',
                    type=str,
                    dest='group_cols',
                    default=None,
                    metavar='',
                    help='group columns [format: group1,group2,...]')
parser.add_argument('-t', '--truncate-mode',
                    type=str,
                    dest='truncate',
                    default='none',
                    metavar='',
                    help='truncate mode [none (default), lastp, level]')
parser.add_argument('-T', '--truncate-parameter',
                    type=int,
                    dest='truncate_p',
                    default=10,
                    metavar='',
                    help='parameter p for truncate mode [default: 10]')
parser.add_argument('-s', '--subset-groups',
                    type=str,
                    dest='subset',
                    default=None,
                    metavar='',
                    help='subset samples [format: column,set1,set2,...]')
parser.add_argument('-S', '--fig-size',
                    type=str,
                    dest='fig_size',
                    default='10x10',
                    metavar='',
                    help='figure size [format: <width>x<height>]')
parser.add_argument('-H', '--hide-statistics',
                    dest='hide_stats',
                    action='store_false',
                    help='do not print statistics [default: False]')
parser.add_argument('-P', '--plot-dendrogram',
                    dest='plot_dendro',
                    action='store_true',
                    help='plot dendrogram [default: False]')
args = parser.parse_args()

# Analysis --------------------------------------------------------------------

# Read distance matrix
dist = pd.read_csv(args.input, sep='\t', index_col=0)

# Remove NAs
dist = cluster_profiles.remove_na(dist, threshold=args.remove_na_threshold)

# Add metadata (if appliable)
if args.metadata_file is not None:
    dist = cluster_profiles.add_metadata(dist,
                                         args.metadata_file,
                                         args.metadata_id)

# Subset groups (if applicable)
if args.subset is not None:

    # Loop over subset groups
    for sub_group in args.subset.split(";"):

        # Get column and sets
        sub_col = sub_group.split(',')[0]
        sub_sets = sub_group.split(',')[1:]
        sub_sets = ','.join(sub_sets)

        # Subset groups
        dist = cluster_profiles.filter_metadata(dist,
                                                filter_col=sub_col,
                                                filter_values=sub_sets)

# BC dataset: remove low quality samples (if applicable)
if 'low_quality' in args.filter:
    dist = cluster_profiles.filter_metadata(dist,
                                            filter_col='sample_quality',
                                            filter_values='ok')

# BC dataset: remove bulk samples (if applicable)
if 'bulk' in args.filter:
    dist = cluster_profiles.filter_metadata(dist,
                                            filter_col='sample_type',
                                            keep_values='single_cell')

# Add groups to index for clustering (if applicable)
dist, k = cluster_profiles.set_index(dist, args.group_cols)

# Perform hierarchical clustering and plot
cluster_profiles.cluster_hierarchical(dist,
                                      args.output,
                                      args.linkage_method,
                                      print_statistics=args.hide_stats,
                                      plot_dendrogram=args.plot_dendro,
                                      truncate_mode=args.truncate,
                                      fig_size=args.fig_size,
                                      p=args.truncate_p,
                                      k=k,
                                      ct=args.colour_threshold)
