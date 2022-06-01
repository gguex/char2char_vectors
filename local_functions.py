import string
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.stats import rankdata
import scipy
from itertools import combinations
from sklearn import linear_model


def process_text(text):
    """
    A function that take a string and remove punctuation, remove numbers, lower cases and remove extra spaces
    :param text: the input string
    :return: the output string
    """
    # Punctuation list
    enhanced_punctuation = string.punctuation + "«»“”—’\n\t"
    # Remove some punctuation
    enhanced_punctuation = re.sub("['-]", "", enhanced_punctuation)
    # Lower char
    processed_text = text.lower()
    # Remove numbers
    processed_text = re.sub(r"\d", " ", processed_text)
    # Remove punctuation
    processed_text = processed_text.translate(str.maketrans(enhanced_punctuation, " " * len(enhanced_punctuation)))
    # Remove special punctuation
    processed_text = re.sub("[-'–][^a-z]", " ", processed_text)
    processed_text = re.sub("[^a-z][-'–]", " ", processed_text)
    # Remove extra spaces
    processed_text = re.sub(" +", " ", processed_text).strip()
    # Return the sentence
    return processed_text


def build_interactions(character_occurrences_df, max_interaction_degree):
    """
    Build interactions up to a certain degree from a character occurrences dataframe.
    :param character_occurrences_df: a pandas dataframe containing character occurrences along unit.
    Columns name should be the names of the characters.
    :param max_interaction_degree: an integer >= 2 indicating the maximum degree of interaction to compute.
    :return: a pandas dataframe containing interactions as columns, unit as row,
    and in the cells the indicator variable of the presence of the interaction in the unit.
    """

    # Get characters
    characters = np.array(character_occurrences_df.columns)
    character_occurrences = character_occurrences_df.to_numpy()

    # Make the list of interactions
    interactions = []
    for id_row in range(character_occurrences.shape[0]):
        presence_in_unit = characters[character_occurrences[id_row, :] > 0]
        for interaction_degree in range(2, max_interaction_degree + 1):
            interactions.extend([tuple(sorted(comb)) for comb in combinations(presence_in_unit, interaction_degree)])
    interactions = list(set(interactions))

    # Fill the interaction presences
    interaction_presences = []
    for interaction in interactions:
        interaction_presence = np.ones(character_occurrences[:, 0].shape) * 1e50
        for char in interaction:
            interaction_presence = np.minimum(interaction_presence,
                                              character_occurrences[:, characters == char].reshape(-1))
        interaction_presences.append(interaction_presence.reshape(-1).astype(int))
    interaction_presences = np.array(interaction_presences).T

    # Build the dataframe of interaction and return it
    interactions_name = ["-".join(interaction) for interaction in interactions]

    return pd.DataFrame(interaction_presences, columns=interactions_name)


def build_directed_interactions(speaking_characters, character_presences_df, max_interaction_degree):
    """
    :param speaking_characters: a list of character speaking to other characters.
    :param character_presences_df: a pandas dataframe containg character occurences along unit. Columns name should be
    the names of the characters
    :param max_interaction_degree: an integer >= 2 indicating the maximum degree of interaction to compute
    :return: a pandas dataframe containing interactions as columns, unit as row,
    and in the cells the indicator variable of the presence of the interaction in the unit
    """

    # Get characters
    characters = np.array(character_presences_df.columns)
    character_presences = character_presences_df.to_numpy()

    # Make the list of interactions
    interactions = []
    for id_row in range(character_presences.shape[0]):
        speaking_character = speaking_characters[id_row]
        presence_in_unit = list(characters[character_presences[id_row, :] > 0])
        if speaking_character in presence_in_unit:
            presence_in_unit.remove(speaking_character)
        for interaction_degree in range(1, max_interaction_degree):
            interactions.extend([tuple([speaking_character] + sorted(comb))
                                 for comb in combinations(presence_in_unit, interaction_degree)])
    interactions = list(set(interactions))

    # Fill the interaction presences
    interaction_presences = []
    for interaction in interactions:
        speaking_character_presence = (np.array(speaking_characters) == interaction[0]) * 1
        for char in interaction[1:]:
            speaking_character_presence = \
                speaking_character_presence * character_presences[:, characters == char].reshape(-1)
        interaction_presences.append(speaking_character_presence.reshape(-1).astype(int))
    interaction_presences = np.array(interaction_presences).T

    # Build the dataframe of interaction and return it
    interactions_name = ["-".join(interaction) for interaction in interactions]

    return pd.DataFrame(interaction_presences, columns=interactions_name)


def sorted_eig(matrix, dim_max=None):
    """
    A function to compute, and sort real part of eigenvalues and eigenvector to dim_max
    :param matrix: the matrix to decompose
    :param dim_max: the maximum number of dimensions
    :return: eigen_values and eigen_vectors arrays
    """

    # Choice of computation depending of dim_max
    if (dim_max is not None) and dim_max < matrix.shape[0] - 1:
        eigen_values, eigen_vectors = scipy.sparse.linalg.eigs(matrix, dim_max)
    else:
        eigen_values, eigen_vectors = scipy.linalg.eigs(matrix)

    # Sort the values
    sorted_indices = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    # Return
    return eigen_values, eigen_vectors


def correspondence_analysis(contingency):
    """
    A function to perform a correspondence analysis from a contingency table

    :param contingency: a contingency table in array or numpy array format
    :return: maximum number of dimension, percentage of variance, row coordinates, col coordinates, row contributions
    col contributions, row square cosines, col square cosines.
    """

    # Transform into a numpy array
    contingency = np.array(contingency)
    # Extract the number of rows and columns
    n_row, n_col = contingency.shape
    # Compute the maximum number of dimension
    dim_max = min(n_row, n_col) - 1

    # Compute the total in contingency table
    total = np.sum(contingency)
    # Compute the row and columns weights
    f_row = contingency.sum(axis=1)
    f_row = f_row / sum(f_row)
    f_col = contingency.sum(axis=0)
    f_col = f_col / sum(f_col)
    # Compute the independence table
    independency = np.outer(f_row, f_col) * total
    # Compute the quotient matrix
    normalized_quotient = contingency / independency - 1

    # Compute the scalar products matrix
    b_mat = (normalized_quotient * f_col) @ normalized_quotient.T
    # Compute the weighted scalar products matrix
    k_mat = np.outer(np.sqrt(f_row), np.sqrt(f_row)) * b_mat
    # Perform the eigen-decomposition
    eig_val, eig_vec = sorted_eig(k_mat)
    # Cut to the maximum of dimensions
    eig_val = eig_val[:dim_max]
    eig_vec = eig_vec[:, :dim_max]

    # Compute the percentage of variance
    percentage_var = eig_val / sum(eig_val)
    # Compute row and col coordinates
    row_coord = np.real(np.outer(1 / np.sqrt(f_row), np.sqrt(eig_val)) * eig_vec)
    col_coord = (normalized_quotient.T * f_row) @ row_coord / np.sqrt(eig_val)
    # Compute row and col contributions
    row_contrib = eig_vec ** 2
    col_contrib = np.outer(f_col, 1 / eig_val) * col_coord ** 2
    # Compute row and col cosines
    row_cos2 = row_coord ** 2
    row_cos2 = (row_cos2.T / row_cos2.sum(axis=1)).T
    col_cos2 = col_coord ** 2
    col_cos2 = (col_cos2.T / col_cos2.sum(axis=1)).T

    return dim_max, percentage_var, row_coord, col_coord, row_contrib, col_contrib, row_cos2, col_cos2


def explore_correspondence_analysis(row_names, col_names, dim_max, row_coord, col_coord,
                                    row_contrib, col_contrib, row_cos2, col_cos2):
    """
    Give some dataframe in order to explore correspondence analysis results
    :param row_names: the names of rows
    :param col_names: the names of columns
    :param dim_max: the maximum number of dimension
    :param row_coord: the coordinates of rows
    :param col_coord: the coordinates of columns
    :param row_contrib: the contributions of rows
    :param col_contrib: the contributions of columns
    :param row_cos2: the square cosine of rows
    :param col_cos2: the square cosine of columns
    :return: multiple df to explore the results.
    """

    # Completed dimension name
    dim_names = [(len(f"{dim_max}") - len(f"{dim}")) * "0" + f"{dim}" for dim in range(dim_max)]

    # --- For rows

    # Make the different value in df
    row_coord_df = pd.DataFrame(row_coord, index=row_names, columns=[dim_name + "_coord" for dim_name in dim_names])
    row_contrib_df = pd.DataFrame(row_contrib, index=row_names, columns=[dim_name + "_contrib"
                                                                         for dim_name in dim_names])
    row_cos2_df = pd.DataFrame(row_cos2, index=row_names, columns=[dim_name + "_cos2" for dim_name in dim_names])
    # Concatenate them, then sort them
    row_explore_df = pd.concat([row_coord_df, row_contrib_df, row_cos2_df], axis=1)
    row_explore_df = row_explore_df.reindex(sorted(row_explore_df.columns), axis=1)

    # Explore cos2
    row_cos2_explore_df = pd.DataFrame(row_cos2.T, columns=row_names)

    # --- For columns

    # Make the different value in df
    col_coord_df = pd.DataFrame(col_coord, index=col_names, columns=[dim_name + "_coord" for dim_name in dim_names])
    col_contrib_df = pd.DataFrame(col_contrib, index=col_names, columns=[dim_name + "_contrib"
                                                                         for dim_name in dim_names])
    col_cos2_df = pd.DataFrame(col_cos2, index=col_names, columns=[dim_name + "_cos2" for dim_name in dim_names])
    # Concatenate them, then sort them
    col_explore_df = pd.concat([col_coord_df, col_contrib_df, col_cos2_df], axis=1)
    col_explore_df = col_explore_df.reindex(sorted(col_explore_df.columns), axis=1)

    # Explore cos2
    col_cos2_explore_df = pd.DataFrame(col_cos2.T, columns=col_names)

    # Return results
    return row_explore_df, row_cos2_explore_df, col_explore_df, col_cos2_explore_df


def build_occurrences_vectors(occurrences, vectors):
    """
    Build the vectors of objects regarding their occurrences in the text.
    :param occurrences: the (units x objects) matrix containing occurrences of objects in units.
    :param vectors: the (units x dim) matrix containing units vectors with dim dimensions.
    :return: object_vectors: a (objects x dim) dataframe containing vectors of objects.
    """

    # Build weighted occurrences
    weighted_occurrences = occurrences / occurrences.sum(axis=0)

    # Compute coordinates
    coord_object = weighted_occurrences.T @ vectors

    # Return them
    return coord_object


def build_regression_vectors(occurrences, vectors, weights, reg_type="Ridge", regularization_parameter=1):
    """
    Build the vectors of objects regarding their regression coefficient with respect to the vectors.
    :param occurrences: the (units x objects) matrix containing occurrences of objects in units.
    :param vectors: the (units x dim) matrix containing units vectors with dim dimensions.
    :param weights: the (units) vector containing unit weights.
    :param reg_type: the type of regression, between "Linear", "Ridge", or "Lasso" (default = "Ridge")
    :param regularization_parameter: the regularization parameter for the regression (default = 1)
    :return: regression_vectors: a (objects x dim) dataframe containing regression vectors of objects.
    """

    # Get the maximum of dimension
    dim_max = vectors.shape[1]

    # An empty array for results
    regression_vectors = []
    # Loop on dimensions
    for i in range(dim_max):
        # Get coordinates for the dimension
        coordinates = vectors[:, i]
        # Chose the regression
        if reg_type == "Ridge":
            reg = linear_model.Ridge(regularization_parameter)
        elif reg_type == "Lasso":
            reg = linear_model.Lasso(regularization_parameter)
        else:
            reg = linear_model.LinearRegression()
        # Fit the regression
        reg.fit(occurrences, coordinates, sample_weight=weights)
        # Store the intercept and the coefficients
        regression_vector = np.concatenate([[reg.intercept_], reg.coef_])
        # Append the results to the result array
        regression_vectors.append(regression_vector)

    # Return the resulting vectors
    return np.array(regression_vectors).T


def display_char_network(interact_list, edge_polarity_list, edge_weight_list, color="polarity", width="weight",
                         width_rank=False, node_min_width=50, node_max_width=1000, edge_min_width=0.2, edge_max_width=5,
                         string_strength="weight", min_alpha=0.5, max_alpha=1, cmap=plt.cm.coolwarm,
                         node_pos=None):
    """
    A function to diplay a graph between character interactions

    :param width_rank:
    :param min_alpha:
    :param interact_list:
    :param edge_polarity_list:
    :param edge_weight_list:
    :param color:
    :param width:
    :param node_min_width:
    :param node_max_width:
    :param edge_min_width:
    :param edge_max_width:
    :param string_strength:
    :param max_alpha:
    :param cmap:
    :param node_pos:
    :return:
    """

    # Make sure quantities are in numpy array
    edge_polarity_list = np.array(edge_polarity_list)
    edge_weight_list = np.array(edge_weight_list)

    # --- Making the graph

    # Make the two vectors of interactions
    from_char_list = []
    to_char_list = []
    for act_interact in interact_list:
        from_char_list.append(act_interact[0])
        to_char_list.append(act_interact[1])

    # Make the graph dict
    graph_dict = {"char_from": from_char_list, "char_to": to_char_list,
                  "polarity": edge_polarity_list,
                  "weight": edge_weight_list}
    # Adding string strength
    if string_strength == "polarity":
        graph_dict["string_strength"] = 10 * (edge_polarity_list - min(edge_polarity_list)) \
                                        / (max(edge_polarity_list) - min(edge_polarity_list)) + 0.1
    else:
        graph_dict["string_strength"] = 10 * (edge_weight_list - min(edge_weight_list)) \
                                        / (max(edge_weight_list) - min(edge_weight_list)) + 0.1
    # Making the graph df
    graph_df = pd.DataFrame(graph_dict)

    # Make the graph
    graph = nx.from_pandas_edgelist(graph_df,
                                    source="char_from",
                                    target="char_to",
                                    edge_attr=True,
                                    create_using=nx.DiGraph())

    # --- Color and widths

    # Computing mean polarity and mean weight of nodes
    node_polarity_list, node_weight_list = np.empty(0), np.empty(0)
    for node in graph.nodes:
        char_df = graph_df[graph_df["char_from"] == node]
        node_polarity_list = np.append(node_polarity_list,
                                       sum(char_df["polarity"] * char_df["weight"]) / char_df["weight"].sum())
        node_weight_list = np.append(node_weight_list, char_df["weight"].sum())

    # Computing node and edge widths
    if width == "polarity":
        node_unscaled_width_list = node_polarity_list
        edge_unscaled_width_list = np.array([graph[u][v]["polarity"] for u, v in graph.edges()])
    else:
        node_unscaled_width_list = node_weight_list
        edge_unscaled_width_list = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    if width_rank:
        node_unscaled_width_list = rankdata(node_unscaled_width_list)
        edge_unscaled_width_list = rankdata(edge_unscaled_width_list)
    node_lambda = (node_unscaled_width_list - min(node_unscaled_width_list)) / \
                  (max(node_unscaled_width_list) - min(node_unscaled_width_list))
    edge_lambda = (edge_unscaled_width_list - min(edge_unscaled_width_list)) / \
                  (max(edge_unscaled_width_list) - min(edge_unscaled_width_list))
    node_width_list = (1 - node_lambda) * node_min_width + node_lambda * node_max_width
    edge_width_list = (1 - edge_lambda) * edge_min_width + edge_lambda * edge_max_width

    # Computing node and edge colors
    if color == "polarity":
        node_color_list = node_polarity_list
        edge_color_list = [graph[u][v]["polarity"] for u, v in graph.edges()]
        edge_alpha_lambda = (np.array(edge_color_list) - min(edge_color_list)) / \
                            (max(edge_color_list) - min(edge_color_list))
        edge_alpha_list = (1 - edge_alpha_lambda) * min_alpha + edge_alpha_lambda * max_alpha
    else:
        node_color_list = node_weight_list
        edge_color_list = [graph[u][v]["weight"] for u, v in graph.edges()]
        edge_alpha_list = 1

    # --- Plotting the graph

    # the positions of nodes
    if node_pos is None:
        node_pos = nx.spring_layout(graph, weight="string_strength")

    # Starting the graph
    plt.figure()

    # Create the nodes
    nx.draw_networkx_nodes(graph,
                           node_size=node_width_list,
                           pos=node_pos,
                           label=from_char_list,
                           node_color=node_color_list,
                           cmap=cmap,
                           vmin=min(edge_color_list),
                           vmax=max(edge_color_list),
                           alpha=0.8)
    # Draw labels
    nx.draw_networkx_labels(graph, pos=node_pos, font_size=25, font_weight="bold")
    # Create the edges
    edges = nx.draw_networkx_edges(graph,
                                   pos=node_pos,
                                   arrows=True,
                                   width=edge_width_list,
                                   node_size=node_width_list,
                                   edge_color=edge_color_list,
                                   edge_cmap=cmap,
                                   edge_vmin=min(edge_color_list),
                                   edge_vmax=max(edge_color_list),
                                   alpha=edge_alpha_list,
                                   connectionstyle="arc3,rad=0.2",
                                   arrowsize=40)

    pc = PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_color_list)

    ax = plt.gca()
    plt.colorbar(pc, ax=ax)
    ax.set_axis_off()
    plt.show()

    return node_pos
