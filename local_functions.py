import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl


def display_char_network(interact_list, edge_polarity_list, edge_weight_list, color="polarity", width="weight",
                         node_min_width=50, node_max_width=1000, edge_min_width=0.2, edge_max_width=5,
                         string_strength="weight", min_alpha=0.6, max_alpha=1, cmap=mpl.pyplot.cm.coolwarm):
    """
    A function to diplay a graph between character interactions

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
    G = nx.from_pandas_edgelist(graph_df,
                                source="char_from",
                                target="char_to",
                                edge_attr=True,
                                create_using=nx.DiGraph())

    # --- Color and widths

    # Computing mean polarity and mean weight of nodes
    node_polarity_list, node_weight_list = np.empty(0), np.empty(0)
    for node in G.nodes:
        char_df = graph_df[graph_df["char_from"] == node]
        node_polarity_list = np.append(node_polarity_list,
                                       sum(char_df["polarity"] * char_df["weight"]) / char_df["weight"].sum())
        node_weight_list = np.append(node_weight_list, char_df["weight"].sum())

    # Computing node and edge widths
    if width == "polarity":
        node_lambda = (node_polarity_list - min(node_polarity_list)) / \
                      (max(node_polarity_list) - min(node_polarity_list))
        edge_unscaled_width_list = np.array([G[u][v]["polarity"] for u, v in G.edges()])
    else:
        node_lambda = (node_weight_list - min(node_weight_list)) / \
                      (max(node_weight_list) - min(node_weight_list))
        edge_unscaled_width_list = np.array([G[u][v]["weight"] for u, v in G.edges()])

    node_width_list = (1 - node_lambda) * node_min_width + node_lambda * node_max_width
    edge_lambda = (edge_unscaled_width_list - min(edge_unscaled_width_list)) / \
                  (max(edge_unscaled_width_list) - min(edge_unscaled_width_list))
    edge_width_list = (1 - edge_lambda) * edge_min_width + edge_lambda * edge_max_width

    # Computing node and edge colors
    if color == "polarity":
        node_color_list = node_polarity_list
        edge_color_list = [G[u][v]["polarity"] for u, v in G.edges()]
        edge_alpha_lambda = (np.array(edge_color_list) - min(edge_color_list)) / \
                            (max(edge_color_list) - min(edge_color_list))
        edge_alpha_list = (1 - edge_alpha_lambda) * min_alpha + edge_alpha_lambda * max_alpha
    else:
        node_color_list = node_weight_list
        edge_color_list = [G[u][v]["weight"] for u, v in G.edges()]

    # --- Plotting the graph

    # the positions of nodes
    node_pos = nx.spring_layout(G, weight="string_strength")

    # Create the nodes
    nodes = nx.draw_networkx_nodes(G,
                                   node_size=node_width_list,
                                   pos=node_pos,
                                   label=from_char_list,
                                   node_color=node_color_list,
                                   cmap=cmap,
                                   vmin=min(edge_color_list),
                                   vmax=max(edge_color_list),
                                   alpha=0.8)
    # Draw labels
    labels = nx.draw_networkx_labels(G, pos=node_pos, font_size=10)
    # Create the edges
    edges = nx.draw_networkx_edges(G,
                                   pos=node_pos,
                                   arrows=True,
                                   width=edge_width_list,
                                   node_size=node_width_list,
                                   edge_color=edge_color_list,
                                   edge_cmap=cmap,
                                   edge_vmin=min(edge_color_list),
                                   edge_vmax=max(edge_color_list),
                                   alpha=edge_alpha_list,
                                   connectionstyle="arc3,rad=0.2")

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_color_list)

    ax = mpl.pyplot.gca()
    mpl.pyplot.colorbar(pc, ax=ax)
    ax.set_axis_off()
    mpl.pyplot.show()