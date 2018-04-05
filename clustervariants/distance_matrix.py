import numpy as np
import pandas as pd


def calculate_distance(data_1,
                       data_2,
                       a=1,
                       b=5):
    """
    Calculates the similarity score distance between two given SNV profiles
    """

    # Merge profiles
    merged = pd.merge(data_1, data_2, on=['chr', 'pos'], how='inner')
    merged = merged.fillna(0)

    # Find number of matches
    matches = merged.loc[merged['genotype_x'] == merged['genotype_y']]
    n_matches = matches.shape[0]

    # Calculate similarity score
    n_total = merged.shape[0]
    distance = 1 - (n_matches + a) / (n_total + a + b)

    # Return distance measure
    return distance


def calculate_distance_matrix(profiles,
                              normalise=True):
    "Calculate distances for all combinations of provided samples"

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
                distance = calculate_distance(profiles[sample1],
                                              profiles[sample2])
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
