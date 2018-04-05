import sklearn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from clustervariants.preprocess_distance_matrix import remove_metadata


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
    # We find the vector parallel to the line by projecting dist_from_first on
    # the line. The perpendicular vector is dist_from_first - parallel_vector
    # We project dist_from_first by taking the scalar product of the vector
    # with the unit vector that points in the direction of the line (this gives
    # us the length of the projection of dist_from_first onto the line). If we
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
