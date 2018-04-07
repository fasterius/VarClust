import os
import pandas as pd


def sort_genotype(row_data):
    "Converts genotypes into 10-element vectors"

    # Sort the genotype column
    sorted_genotype = ''.join(sorted(row_data['genotype']))
    row_data['genotype'] = sorted_genotype

    # Return row_data with sorted genotype
    return row_data


def read_profile(file,
                 by_unique='gene',
                 subset_cols=None,
                 subset_values=None):
    """Reads a SNV profile."""

    # Read file with pandas
    data = pd.read_table(file, sep='\t')

    # Drop non-standard chromosomes
    standard = ['chr' + str(chr) for chr in range(1, 23)]
    standard.extend(['chrX', 'chrY', 'chrMT'])
    data = data.loc[data['chr'].isin(standard)]

    # Subset (if applicable)
    if subset_cols is not None:
        if subset_values is not None:
            for col in subset_cols:
                data = data.loc[data[col].isin(subset_values)]

    # Remove duplicated variants at specified level (if present)
    if by_unique == 'gene':
        unique_level = ['chr', 'pos', 'ENSGID']
    elif by_unique == 'position':
        unique_level = ['chr', 'pos']
    else:
        raise ValueError('unique variant level specification \"' + by_unique +
                         '\" not valid; use \"gene\" or \"position\".')
    data = data.drop_duplicates(subset=unique_level)

    # Merge alleles into genotypes
    data['genotype'] = data['A1'] + data['A2']

    # Sort genotypes
    data = data.apply(sort_genotype, axis=1)

    # Drop allele columns
    data = data.drop(columns=['A1', 'A2'])

    # Return final dataframe
    return data


def read_profile_dir(in_dir,
                     by_unique='gene',
                     samples=None,
                     subset_cols=None,
                     subset_values=None):
    "Reads all SNV profiles in a given directory"

    # List all profile files
    files = [os.path.join(in_dir, file) for file in os.listdir(in_dir)]
    files = [name for name in files if '.profile.txt' in name]

    # Calculate total profiles to read and initialise counter
    nn_tot = min(len(files), len(samples))
    nn = 1

    # Read each profile and save in a dictionary
    profiles = {}
    for file in files:

        # Get current sample name
        sample = file.split('/')[-1].split('.')[0]

        # Only read provided samples (if applicable)
        if samples is not None:
            if sample not in samples:
                continue

        # Read profile and add to dictionary
        print('Reading profile for ' + sample + ' [' + str(nn) + ' / ' +
              str(nn_tot) + ']')
        profiles[sample] = read_profile(file, by_unique, subset_cols,
                                        subset_values)

        # Increment counter
        nn += 1

    # Return profiles
    print('done.')
    return profiles
