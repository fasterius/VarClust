#!/usr/bin/env python3

# Import modules
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import cluster_profiles

# Argument parser -------------------------------------------------------------

parser = argparse.ArgumentParser(epilog='Clusters and plots a distance matrix '
                                        'using the tSNE clustering method.')
parser.add_argument('input',
                    type=str,
                    help='input distance matrix file path')
parser.add_argument('output',
                    type=str,
                    help='output image file path')
parser.add_argument('-r', '--remove-na-threshold',
                    type=int,
                    default=5,
                    dest='remove_na_threshold',
                    metavar='',
                    help='remove rows/columns containing NAs >= threshold')
parser.add_argument('-a', '--alpha',
                    type=float,
                    default=0.75,
                    dest='alpha',
                    metavar='',
                    help='alpha value for points [default: 0.75]')
parser.add_argument('-m', '--metadata-file',
                    type=str,
                    dest='metadata_file',
                    default=None,
                    metavar='',
                    help='metadata file path')
parser.add_argument('-i', '--metadata-id',
                    type=str,
                    dest='metadata_id',
                    default=None,
                    metavar='',
                    help='metadata id column to merge on')
parser.add_argument('-f', '--filter',
                    type=str,
                    dest='filter',
                    default='none',
                    metavar='',
                    help='filter cells [none (default), low_quality, '
                         'low_quality_and_bulk]')
parser.add_argument('-c', '--colour-col',
                    type=str,
                    dest='colour_col',
                    default=None,
                    metavar='',
                    help='grouping column for colours [default: none]')
parser.add_argument('-s', '--shape-col',
                    type=str,
                    dest='shape_col',
                    default=None,
                    metavar='',
                    help='grouping column for shapes [default: none]')
parser.add_argument('-S', '--subset-groups',
                    type=str,
                    dest='subset',
                    default=None,
                    metavar='',
                    help='subset samples [format: col1,set1,set2,;col2,...]')
parser.add_argument('-p', '--perplexity',
                    type=int,
                    dest='perplexity',
                    default=30,
                    metavar='',
                    help='perplexity [default: 30]')
parser.add_argument('-l', '--learning-rate',
                    type=int,
                    dest='learning_rate',
                    default=20,
                    metavar='',
                    help='learning rate [default: 200]')
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
#  groups = ','.join([args.colour_col, args.shape_col])
#  dist, k = cluster_profiles.set_index(dist, groups)

# Perform tSNE
tsne = cluster_profiles.cluster_tsne(dist=dist,
                                     learning_rate=args.learning_rate,
                                     perplexity=args.perplexity)

# Plot tSNE
plot = cluster_profiles.plot_tsne(tsne=tsne,
                                  dist=dist,
                                  output=args.output,
                                  alpha=args.alpha,
                                  colour_cols=args.colour_col,
                                  shape_cols=args.shape_col)
