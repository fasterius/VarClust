#!/usr/bin/env python3

# Import modules
import seaborn as sns
import sklearn
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.colors import rgb2hex
from scipy.cluster.hierarchy \
    import dendrogram, linkage, set_link_color_palette, fcluster


def remove_na(dist, threshold=5):
    "Removes combinations containing NA values from a distance matrix."

    # Find and remove IDs for samples containing only NAs
    nas = dist[dist.sum(axis=0) == 0].index.tolist()
    dist = dist.drop(axis=0, labels=nas)
    dist = dist.drop(axis=1, labels=nas)

    # Find and remove IDs for rows/columns containing NAs above threshold
    nas = dist[dist.isnull().sum(axis=1) >= threshold].index.tolist()
    dist = dist.drop(axis=0, labels=nas)
    dist = dist.drop(axis=1, labels=nas)

    # Return distance matrix
    return dist


def add_metadata(distances,
                 metadata_file,
                 id_col):
    "Adds metadata to a distance matrix."

    if id_col is None:
        raise RuntimeError('metadata id column missing.')

    # Read metadata
    metadata = pd.read_table(metadata_file, encoding='iso8859_16')

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
                    filter_values):
    "Filter a distance matrix on its metadata columns."

    # Check if column exists in data
    if filter_col not in dist.columns:
        raise RuntimeError(filter_col + " column not in data")

    # Get indexes to drop
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


def remove_metadata(dist):
    "Removes metadata columns from a distance matrix."

    # Check if distance matrix contains metadata and remove if applicable
    if dist.shape[0] != dist.shape[1]:
        dist = dist.iloc[:, 0:dist.shape[0]]

    # Return pure distance matrix
    return dist


def find_best_k(distances,
                max_k=20):
    "Finds the optimal number of cluster using k-means with the elbow method."

    # Calculate errors per k
    cluster_errors = []
    cluster_range = range(1, min(max_k, distances.shape[0]))
    for n_clusters in cluster_range:
        clusters = KMeans(n_clusters)
        clusters.fit(distances)
        cluster_errors.append(clusters.inertia_)

    # Get coordinates of all the points
    n_points = len(cluster_errors)
    coords = np.vstack((range(1, n_points + 1), cluster_errors)).T
    first_point = coords[0]

    # Get vector between first and last point - this is the line
    line_vector = coords[-1] - coords[0]

    # normalize the line vector
    line_norm = line_vector / np.sqrt(np.sum(line_vector**2))

    # find the distance from each point to the line:
    # vector between all points and first point
    dist_from_first = coords - first_point

    # To calculate the distance to the line, we split dist_from_first into two
    # components, one that is parallel to the line and one that is
    # perpendicular. Then, we take the norm of the part that is perpendicular
    # to the line and get the distance.
    # We find the vector parallel to the line by projecting dist_from_first onto
    # the line. The perpendicular vector is dist_from_first - parallel_vector
    # We project dist_from_first by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of dist_from_first onto the line). If we
    # multiply the scalar product by the unit vector, we have
    # parallel_vector
    scalar_product = np.sum(dist_from_first * np.matlib.repmat(line_norm,
                                                               n_points, 1),
                            axis=1)
    parallel_vector = np.outer(scalar_product, line_norm)
    vector_to_line = dist_from_first - parallel_vector

    # distance to line is the norm of vector_to_line
    dist_to_line = np.sqrt(np.sum(vector_to_line ** 2, axis=1))

    # now all you need is to find the maximum
    idx_best = np.argmax(dist_to_line)
    best_k = int(coords[idx_best, 0])
    best_coord_x = coords[idx_best, 0]
    best_coord_y = coords[idx_best, 1]

    # Plot
    data = pd.DataFrame(cluster_errors, index=cluster_range, columns=['error'])
    plt.figure()
    plt.plot(data, linestyle='-', marker='o', markersize=4)
    plt.xlabel('k')
    plt.ylabel('Sum-of-squared error')
    plt.scatter(x=best_coord_x, y=best_coord_y, color='black', s=60)
    plt.annotate('k = ' + str(best_k), xy=(best_coord_x + 0.5,
                                           best_coord_y + 0.5))
    plt.savefig("elbow.png")

    # Return the best k value
    return best_k


def cluster_hierarchical(distances,
                         output,
                         method='complete',
                         truncate_mode='none',
                         print_statistics=True,
                         plot_dendrogram=False,
                         fig_size="10x10",
                         p=10,
                         k=1,
                         ct=0):
    "Cluster a distance matric with hierarchical clustering."

    # Remove metadata
    distances = remove_metadata(distances)

    # Calcualate linkage
    linkages = linkage(dist.squareform(distances), method=method)

    # Get true labels
    cells_true = [index.split(': ')[0] for index in distances.index]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(cells_true)
    labels_true = le.transform(cells_true)

    # Calculate ARI
    labels_pred = fcluster(linkages, t=k, criterion='maxclust')
    ari = sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
    ari = round(ari, 2)

    # Print statistics (if applicable)
    if print_statistics:

        # Gather statistics
        n_samples = str(len(distances))
        out = output.split('/')[-1].replace('dendrogram.', '').split('.')
        out_dataset = out[0]
        out_metric = out[1]
        out_merge = out[2]
        out_subset = out[3]
        out_group = out[4]
        out_filter = out[5]

        # Print statistics to STDOUT
        out_string = out_dataset + '\t' + \
            n_samples + '\t' + \
            out_metric + '\t' + \
            out_merge + '\t' + \
            out_subset + '\t' + \
            out_group + '\t' +\
            out_filter + '\t' + \
            method + '\t' + \
            str(ari)
        print(out_string)

    # Plot dendrogram
    if plot_dendrogram:

        # Get groups from index for colouring
        colours = pd.DataFrame(distances.index)
        colours.columns = ['index']
        colours['label'] = colours['index'].str.split(': ', 1).str[0]
        colours_index = colours.set_index('index')

        # Map colours to groups
        sns.set_palette('tab20', k, 0.65)
        palette = sns.color_palette()
        groups = colours['label'].sort_values().unique()
        colour_map = dict(zip(groups, palette))
        colours = colours_index
        colours['group'] = colours['label'].map(colour_map)
        colours = colours.drop(axis=1, labels='label')

        # Find the best k to use and add as additional row colours
        best_k = find_best_k(distances)
        #  best_k = 11
        labels_k = fcluster(linkages, t=best_k, criterion='maxclust')
        sns.set_palette('Greys', best_k, 1)
        palette_k = sns.color_palette()
        groups_k = set(labels_k)
        colour_map_k = dict(zip(groups_k, palette_k))
        colours_k = colours_index
        colours_k['label'] = labels_k
        colours_k['cluster'] = colours_k['label'].map(colour_map_k)
        colours_k = colours_k.drop(axis=1, labels='label')

        # Combine both row colours
        row_colours = [colours_k['cluster'], colours['group']]

        # Plot figure
        cp = sns.clustermap(distances,
                            row_linkage=linkages,
                            col_linkage=linkages,
                            xticklabels=False,
                            yticklabels=False,
                            row_colors=row_colours,
                            cmap='Blues_r',
                            z_score=0,
                            robust=True)
        cp.ax_col_dendrogram.set_visible(False)

        # Set colour gradient legend position and title
        cp.cax.set_position([0.915, .2, .03, .45])
        cp.cax.set_title('Z-score', loc='left')

        # Add colour legend for groups
        #  all_labels = groups.tolist() + list(groups_k)
        all_labels = groups
        for label in all_labels:
            try:
                current_colour = colour_map[label]
            except KeyError:
                current_colour = colour_map_k[label]
                label = 'Cluster ' + str(label)
            cp.ax_heatmap.bar(0, 0,
                              color=current_colour,
                              label=label,
                              linewidth=0)
        cp.ax_heatmap.legend(loc='center',
                             ncol=int(round((k+1)/2)),
                             bbox_to_anchor=(0.5, -0.05))
        cp.ax_heatmap.set_title('ARI = ' + str(ari))

        # Save plot to file
        plt.savefig(output, dpi=300, bbox_inches='tight')


def cluster_tsne(dist,
                 perplexity=30,
                 learning_rate=200):
    "Cluster a distance matrix using tSNE."

    # Remove metadata
    dist = remove_metadata(dist)

    # Perform tSNE
    model = TSNE(n_components=2,
                 metric='precomputed',
                 perplexity=perplexity,
                 learning_rate=learning_rate)
    tsne = model.fit_transform(dist)

    # Convert to dataframe
    tsne = pd.DataFrame(tsne)
    tsne.columns = ['x', 'y']

    # Return tSNE resuts
    return tsne


def create_tsne_plot(tsne,
                     output,
                     alpha=0.75):
    "Plot results from tSNE clustering."

    # Plot setup
    fig, ax = plt.subplots()
    ax.set_xlim(min(tsne['x']) * 1.1, max(tsne['x']) * 1.1)
    ax.set_ylim(min(tsne['y']) * 1.1, max(tsne['y']) * 1.1)
    ax.set(xlabel='tSNE dimension 1', ylabel='tSNE dimension 2')

    # Get colour labels and map to groups
    unique_colours = tsne['colour'].unique()
    colour_groups = sorted([x for x in unique_colours if str(x) != 'nan'])
    rgb_values = sns.color_palette('tab20', len(colour_groups))
    sns.set_palette(rgb_values, n_colors=len(colour_groups))
    colour_map = dict(zip(colour_groups, rgb_values))

    # Get shape labels and map to individual shapes
    possible_shapes = ['.', '*', '^', '8', 's', 'p', '+', 'x', 'D']
    unique_shapes = tsne['shape'].unique()
    shape_groups = sorted([x for x in unique_shapes if str(x) != 'nan'])
    shape_values = possible_shapes[:len(shape_groups)]
    shape_map = dict(zip(shape_groups, shape_values))

    # Loop over shapes
    for shape_group in shape_groups:

        # Subset for current shape
        current_shape = tsne.loc[tsne['shape'] == shape_group, ]

        # Loop over groups
        for colour_group in colour_groups:

            # Get current data
            current_data = current_shape.loc[current_shape['colour'] ==
                                             colour_group, ]

            # Add point borders if current shape is not the first shape
            if shape_group == shape_groups[0]:
                border = None
                label = colour_group
            else:
                border = 'black'
                label = None

            # Plot current shape and group
            ax.scatter(x=current_data['x'],
                       y=current_data['y'],
                       marker=shape_map.get(shape_group),
                       color=colour_map.get(colour_group),
                       edgecolor=border,
                       linewidth=0.25,
                       label=label,
                       alpha=alpha)

    # Fix layout and make space for the legends
    plt.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Add colour legend
    colour_legend = ax.legend(bbox_to_anchor=(1.05, 1),
                              loc=2,
                              borderaxespad=0,
                              fontsize='x-small',
                              markerscale=2)
    ax.add_artist(colour_legend)

    # Manually add marker legend
    if len(shape_groups) > 1:
        marker_legends = []
        for shape in shape_groups:

            # Create patch and add to list
            mleg = plt.scatter([], [],
                               marker=shape_map.get(shape),
                               label=shape,
                               color='black')
            marker_legends.append(mleg)

        # Add manual marker legend
        plt.legend(handles=marker_legends,
                   bbox_to_anchor=(1.05, 0),
                   loc=3,
                   borderaxespad=0,
                   fontsize='x-small')

    # Save plot to file
    plt.savefig(output, dpi=300)


def plot_tsne(tsne,
              dist,
              output,
              alpha=0.75,
              colour_cols=None,
              shape_cols=None):
    "Add metadata to an existing tSNE dataframe and plot for each group."

    # Loop over each supplied colour and shape columns
    for colour_col, shape_col in zip(colour_cols.split(","),
                                     shape_cols.split(",")):

        # Get colour column
        if colour_col.lower() != "none":
            colours = dist[colour_col]
            colour_out = "." + colour_col
        else:
            dist['colour_temp'] = 1
            colours = dist['colour_temp']
            colour_out = ""

        # Get shape column
        if (shape_col.lower() != "none") and (shape_col != colour_col):
            shapes = dist[shape_col]
            shape_out = "." + shape_col
        else:
            dist['shape_temp'] = 1
            shapes = dist['shape_temp']
            shape_out = ""

        # Add colour and shape columns
        tsne['colour'] = colours.values
        tsne['shape'] = shapes.values

        # Add colour/shape names to output
        final_output = output.replace(".png", colour_out + shape_out + ".png")

        # Plot tSNE
        create_tsne_plot(tsne, final_output)
