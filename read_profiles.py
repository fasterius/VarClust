#!/usr/bin/env python3

# Import modules
import os
import pandas as pd


def vectorise(data):
    "Converts genotypes into 10-element vectors"

    # Define 10-element genotype vector
    geno = ['AA', 'AC', 'AG', 'AT', 'CC', 'CG', 'CT', 'GG', 'GT', 'TT']

    # Sort the genotype column
    sorted_genotype = ''.join(sorted(data['genotype']))
    data['genotype'] = sorted_genotype

    # Find index of (sorted) genotypes
    index = 10 - geno.index(sorted_genotype)

    # Add 1 to relevant column
    data[-index] = 1

    # Return data with sorted genotype and vectorised genotype columns
    return data


def read_profile(file,
                 subset_col=None,
                 subset_values=None):
    """Reads an SNV profile created by seqCAT and converts each genotype into
    10-element one-hot vectors."""

    # Read file with pandas
    data = pd.read_table(file, sep='\t')

    # Drop non-standard chromosomes
    standard = ['chr' + str(chr) for chr in range(1, 23)]
    standard.extend(['chrX', 'chrY', 'chrMT'])
    data = data.loc[data['chr'].isin(standard)]

    # Subset (if applicable)
    if subset_col is not None:
        if subset_values is not None:
            for col in subset_col:
                data = data.loc[data[col].isin(subset_values)]

    # Remove duplicated rows (if present)
    data = data.drop_duplicates(subset=['chr', 'pos'])

    # Merge alleles into genotypes
    data['genotype'] = data['A1'] + data['A2']

    # Define 10-element genotype vector and add to dataframe
    geno = ['AA', 'AC', 'AG', 'AT', 'CC', 'CG', 'CT', 'GG', 'GT', 'TT']
    data = data.reindex(columns=list(data) + geno, fill_value=0)

    # Vectorise genotypes
    data = data.apply(vectorise, axis=1)

    # Keep relevant columns
    data = data[['chr', 'pos', 'genotype'] + geno]

    # Return final dataframe
    return data


def read_profile_dir(in_dir,
                     subset_col=None,
                     subset_values=None):
    "Reads all SNV profiles in a given directory"

    # List all profile files
    files = [os.path.join(in_dir, file) for file in os.listdir(in_dir)]
    files = [name for name in files if '.profile.txt' in name]

    # Calculate total profiles to read and initialise counter
    nn_tot = len(files)
    nn = 1

    # Read each profile and save in a dictionary
    profiles = {}
    for file in files:

        # Get current sample name
        sample = file.split('/')[-1].split('.')[0]

        # Read profile and add to dictionary
        print('Reading profile for ' + sample + ' [' + str(nn) + ' / ' +
              str(nn_tot) + ']')
        profiles[sample] = read_profile(file, subset_col, subset_values)

        # Increment counter
        nn += 1

    # Return profiles
    print('done.')
    return profiles
