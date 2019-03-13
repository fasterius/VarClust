import numpy as np
import pandas as pd
import editdistance


def calculate_distance(data_1,
                       data_2,
                       metric='similarity_score',
                       merge_by='gene',
                       a=1,
                       b=5):
    """
    Calculates the similarity score distance between two given SNV profiles
    """

    # Check merge level
    if merge_by == 'gene':
        merge_level = ['chr', 'pos', 'ENSGID']
    elif merge_by == 'position':
        merge_level = ['chr', 'pos']
    else:
        raise ValueError('invalid `merge_by` specification \"' + merge_by +
                         '\"; please use \"gene\" or \"position\".')

    # Merge profiles
    merged = pd.merge(data_1, data_2, on=merge_level, how='inner')
    merged = merged.fillna(0)

    # Calculate distance matrix using supplied metric
    if metric == 'similarity_score' or metric == 'concordance':

        # Find number of matches and total variants
        matches = merged.loc[merged['genotype_x'] == merged['genotype_y']]
        n_matches = matches.shape[0]
        n_total = merged.shape[0]

        # Set a and b to 0 for concordance
        if metric == 'concordance':
            a = b = 0

        # Calculate similarity score or concordance
        distance = 1 - (n_matches + a) / (n_total + a + b)

    elif metric == 'levenshtein':

        # Create per-sample genotype strings
        x = merged[['genotype_x']].to_string(header=False, index=False,
                                             index_names=False).split('\n')
        y = merged[['genotype_y']].to_string(header=False, index=False,
                                             index_names=False).split('\n')
        string_x = ''.join([''.join(element.split()) for element in x])
        string_y = ''.join([''.join(element.split()) for element in y])

        # Calculate edit distance
        distance = editdistance.eval(string_x, string_y)

    else:
        raise ValueError('invalid `metric` specification \"' + metric +
                         '\"; please use \"similarity_score\" or ' +
                         '\"concordance\".')

    # Return distance measure
    return distance


def create_matrix(profiles,
                  metric='similarity_score',
                  merge_by='gene',
                  a=1,
                  b=5,
                  normalise=False):
    "Calculate a distance matrix for all provided samples."

    # Get all samples
    print('Calculating distances ...')
    samples = sorted(list(profiles.keys()))

    # Create an empty distance matrix for all samples
    distances = np.zeros([len(samples), len(samples)])

    # Calculate total number of comparisons and initialise counter
    nn_tot = int(len(samples) * (len(samples) - 1) / 2)
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
                distance = calculate_distance(data_1=profiles[sample1],
                                              data_2=profiles[sample2],
                                              metric=metric,
                                              merge_by=merge_by,
                                              a=a,
                                              b=b)
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
