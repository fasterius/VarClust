import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from varclust.metadata import remove_metadata


def tSNE(dist,
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


def create_tSNE_plot(tsne,
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


def plot_tSNE(tsne,
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
        else:
            dist['colour_temp'] = 1
            colours = dist['colour_temp']

        # Get shape column
        if (shape_col.lower() != "none") and (shape_col != colour_col):
            shapes = dist[shape_col]
        else:
            dist['shape_temp'] = 1
            shapes = dist['shape_temp']

        # Add colour and shape columns
        tsne['colour'] = colours.values
        tsne['shape'] = shapes.values

        # Plot tSNE
        create_tSNE_plot(tsne, output)
