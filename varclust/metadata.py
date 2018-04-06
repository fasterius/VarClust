import pandas as pd


def remove_na(distance, threshold=5):
    "Removes combinations containing NA values from a distance matrix."

    # Find and remove IDs for samples containing only NAs
    nas = distance[distance.sum(axis=0) == 0].index.tolist()
    distance = distance.drop(axis=0, labels=nas)
    distance = distance.drop(axis=1, labels=nas)

    # Find and remove IDs for rows/columns containing NAs above threshold
    nas = distance[distance.isnull().sum(axis=1) >= threshold].index.tolist()
    distance = distance.drop(axis=0, labels=nas)
    distance = distance.drop(axis=1, labels=nas)

    # Return distance matrix
    return distance


def add_metadata(distances,
                 metadata_file,
                 id_col,
                 encoding='iso8859_16'):
    "Adds metadata to a distance matrix."

    if id_col is None:
        raise RuntimeError('metadata id column missing.')

    # Read metadata
    metadata = pd.read_table(metadata_file, encoding=encoding)

    # Merge with distance matrix
    distances[id_col] = distances.index
    merged = pd.merge(distances, metadata, on=id_col)
    merged = merged.drop_duplicates(subset=id_col)

    # Re-add index from ID column
    merged = merged.set_index(id_col)

    # Return distances with metadata
    return merged


def filter_metadata(dist,
                    filter_col,
                    filter_values,
                    expression=False):
    "Filter a distance matrix on its metadata columns."

    # Check if column exists in data
    if filter_col not in dist.columns:
        raise RuntimeError(filter_col + " column not in data")

    # Get indexes to drop
    if expression:
        eval_string = 'dist[filter_col]' + filter_values
        remove_index = dist.loc[~eval(eval_string)].index
    else:
        filter_values = filter_values.split(',')
        remove_index = dist[~dist[filter_col].isin(filter_values)].index

    # Drop the appropriate indexes from rows and columns
    for ax in [0, 1]:
        dist = dist.drop(axis=ax, labels=remove_index, errors='ignore')

    # Return filtered distance matrix
    return dist


def set_index(dist,
              cols_to_add=None):
    """
    Adds metadata to the index of a distance matrix for clustering (if
    applicable) and count the number of unique groups.
    """

    # Add metadata columns to index as groups (if applicable)
    if cols_to_add is not None:

        # Get cols to add
        cols = cols_to_add.split(',')

        # Add column(s) without sample IDs
        dist['group'] = dist[cols[0]]
        if len(cols) > 1:

            for col in cols[1:]:
                dist['group'] = dist['group'] + '.' + dist[col]

        # Get indexes and combine with groups
        dist['index'] = dist.index
        new_index = dist['group'] + ': ' + dist['index']
        dist.index = new_index

        # Count unique groups
        k = len(dist['group'].unique())

    else:

        # Count unique IDs
        k = len(set(dist.index.tolist()))

    # Return distance matrix with new index and k
    return dist, k


def remove_metadata(distance):
    "Removes metadata columns from a distance matrix."

    # Check if distance matrix contains metadata and remove if applicable
    if distance.shape[0] != distance.shape[1]:
        distance = distance.iloc[:, 0:distance.shape[0]]

    # Return pure distance matrix
    return distance
