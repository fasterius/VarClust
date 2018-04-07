import pandas as pd
import varclust.read_profiles as readp


def overlap_profiles(pseudo,
                     profile,
                     merge_by='gene'):
    """
    Overlaps and merge a pseudo-profiles and an SNV profile by the union of
    their matching SNVs.
    """

    # Check that `pseudo` contains a <count> column
    if 'count' not in pseudo.columns:
        raise ValueError('\"count\" column not present in provided ' +
                         'pseudo-profile')

    # Get columns to merge on
    if merge_by == 'gene':
        merge_cols = ['chr', 'pos', 'ENSGID']
    elif merge_by == 'position':
        merge_cols = ['chr', 'pos']
    else:
        raise ValueError('merge level specification \"' + merge_by +
                         '\" invalid; use \"gene\" or \"position\"')

    # Find available meta-columns
    all_cols = set(['rsID', 'gene', 'impact', 'effect', 'feature', 'biotype',
                    'genotype'])
    meta_cols = list(all_cols.intersection(set(pseudo.columns)))

    # Merge profiles
    pseudo = pd.merge(pseudo, profile, on=merge_cols, how='outer')

    # Remove non-matching genotypes
    pseudo = pseudo.loc[(pseudo['genotype_x'] == pseudo['genotype_y']) |
                        pseudo['genotype_x'].isnull() |
                        pseudo['genotype_y'].isnull()]

    # Move profile-unique rows to the pseudo columns
    for col in meta_cols:
        pseudo.loc[pseudo[col + '_x'].isnull(), col + '_x'] = \
            pseudo.loc[pseudo[col + '_x'].isnull(), col + '_y']

    # Increase variant counter
    pseudo.loc[pseudo['count'].isnull(), 'count'] = 0
    pseudo['count'] = pseudo['count'] + 1

    # Remove redundant columns
    pseudo = pseudo[merge_cols + [col + '_x' for col in meta_cols] + ['count']]
    pseudo.columns = merge_cols + meta_cols + ['count']
    pseudo = pseudo.drop_duplicates()

    # Return pseudoprofile
    return pseudo


def create_pseudo_profile(input_dir,
                          output_file,
                          metadata=None,
                          metadata_id_col=None,
                          sample_subset_cols=None,
                          sample_subset_values=None,
                          variant_subset_cols=None,
                          variant_subset_values=None,
                          merge_by='gene',
                          cutoff=10,
                          encoding='iso8859_16'):
    """
    Create a pseudo-profile of all the SNV profiles given in the input
    directory.
    """

    # Read, subset and group on metadata or read all profiles
    if metadata is not None:

        # Check for metadata ID column
        if metadata_id_col is None:
            raise ValueError('Please provide a metadata ID column.')

        # Read metadata
        metadata = pd.read_table(metadata, encoding=encoding)

        # Subset metadata (if applicable)
        if sample_subset_cols is not None and sample_subset_values is not None:
            mask = metadata[sample_subset_cols].isin(sample_subset_values)\
                    .any(axis=1)
            metadata = metadata.loc[mask]
        elif (sample_subset_cols is None and
              sample_subset_values is not None) or \
             (sample_subset_cols is not None and
              sample_subset_values is None):
            raise ValueError('Please provide both column and values to subset '
                             'on if subsetting is desired.')

        # Read profile subset
        sample_subset = metadata[metadata_id_col].tolist()

    else:

        # No sample subsetting
        sample_subset = None

    # Read profiles
    profiles = readp.read_profile_dir(input_dir,
                                      merge_by,
                                      sample_subset,
                                      variant_subset_cols,
                                      variant_subset_values)

    # Initialise pseudo-profile with the first sample
    print('Creating pseudo-profile ...')
    samples = sorted(list(profiles.keys()))
    pseudo = profiles[samples[0]]
    pseudo['count'] = 1

    # Create pseudo-profile
    nn = 1
    for sample in samples[1:len(samples)]:
        print('Overlapping sample ' + str(nn) + ' / ' + str(len(profiles)-1))
        pseudo = overlap_profiles(pseudo, profiles[sample], merge_by)
        nn += 1

    # Add proportions of each variant in final pseudo-profile
    pseudo['proportion'] = round(pseudo['count'] / len(profiles) * 100, 1)

    # Remove variants/genes below proportion cutoff
    pseudo = pseudo.loc[pseudo['proportion'] >= cutoff, ]

    # Write to file
    pseudo.to_csv(output_file, sep='\t', index=False)
    print('Done.')
