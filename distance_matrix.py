#!/usr/bin/env python3

# Load modules
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import itertools
from math import sqrt


def calculate_distance(data1,
                       data2,
                       merge='inner',
                       metric='similarity_score',
                       a=1,
                       b=5,
                       le=1,
                       lc=1):
    """
    Calculates the distance between the two given samples for a given
    metric, either cosine, euclidean or similarity_score.
    """

    # Merge datasets
    merged = pd.merge(data1, data2, on=['chr', 'pos'], how=merge)
    merged = merged.fillna(0)

    # Simple match/mismatch similarity score (if applicable)
    if metric == 'similarity_score':

        # Find number of matches
        matches = merged.loc[merged['genotype_x'] == merged['genotype_y']]
        n_matches = matches.shape[0]

        # Calculate similarity score
        n_total = merged.shape[0]
        distance = 1 - (n_matches + a) / (n_total + a + b)

    else:

        # Remove genotype columns
        geno_cols = list(merged.filter(like='genotype'))
        merged = merged.drop(columns=geno_cols)

        # Separate into sample-specific matrices
        cols = list(merged.filter(like='_x'))
        mat1 = np.array(merged[cols])
        cols = list(merged.filter(like='_y'))
        mat2 = np.array(merged[cols])

        # Collapse into vectors and merge
        vector1 = list(itertools.chain.from_iterable(mat1))
        vector2 = list(itertools.chain.from_iterable(mat2))

        # Calculate the distance with specified metric
        if metric == 'euclidean':
            distance = dist.euclidean(vector1, vector2)
        elif metric == 'cosine':
            distance = dist.cosine(vector1, vector2)
        elif metric == 'correlation':
            distance = dist.correlation(vector1, vector2)

    # Return distance measure
    return distance


def calculate_distance_matrix(profiles,
                              metric='correlation',
                              merge='outer',
                              normalise=True):
    "Calculate distances for all combinations of provided samples"

    # Get all samples
    print('Calculating distances ...')
    samples = sorted(list(profiles.keys()))

    # Create an empty distance matrix for all samples
    distances = np.zeros([len(samples), len(samples)])

    # Calculate total number of comparisons and initialise counter
    nn_tot = int(len(samples) * (len(samples) - 1) / 2) + len(samples)
    nn = 1

    # Calculate distances for each pairwise comparison
    for n in range(len(samples)):
        for m in range(len(samples)):

            # Get samples
            sample1 = samples[n]
            sample2 = samples[m]

            # Skip same-sample comparisons
            if sample1 == sample2:
                continue

            # Skip already compared pairs
            if distances[n, m] != 0 and distances[m, n] != 0:
                continue

            # Calculate distance between current samples
            print('Comparing ' + sample1 + ' and ' + sample2 + ' [' + str(nn) +
                  ' / ' + str(nn_tot) + ']')
            try:
                distance = calculate_distance(profiles[sample1],
                                              profiles[sample2],
                                              metric=metric,
                                              merge=merge)
            except:
                print("Error: " + sample1 + " vs " + sample2)
                continue

            # Add to distance matrix
            distances[n, m] = distance
            distances[m, n] = distance

            # Increment counter
            nn += 1

    # Normalise final distance matrix (if applicable)
    if (normalise):
        distances = distances / distances.max()

    # Convert to dataframe and set row/column indices
    distances = pd.DataFrame(distances, index=samples, columns=samples)

    # Return distance matrix
    print('Done.')
    return distances
