import networkx as nx
import numpy as np


def r(G: nx.Graph, x: int, vi:int , y: int):
    """
    Internal function. Compute r value in dependency computation.

    Args:
        G (nx.Graph): NetworkX undirected graph.
        x (int): Index of x node.
        vi (int): Index of vi node.
        y (int): Index of y node.

    Returns:
        float: r value
    """
    weight_viy = get_edge_weight(G, vi, y)
    weight_xvi = get_edge_weight(G, x, vi)

    return weight_viy / (weight_xvi + weight_viy)


def get_edge_weight(G: nx.Graph, x: int, y: int):
    """
    Internal value. Return edge weight between node x and y. If edge does not exists, return 0.0.

    Args:
        G (nx.Graph): _description_
        x (int): _description_
        y (int): _description_

    Returns:
        float: Edge weight.
    """
    if G.has_edge(x, y):
        return G[x][y]["weight"]
    else:
        return 0.0


def dependency_X_Y(G: nx.Graph, dependency_matrix: np.ndarray, x: int, y:int ) -> float:
    """
    Compute dependency of node X on node Y and store it to dependency_matrix.
    If it is computed, it returns already computed value.

    Args:
        G (nx.Graph): NetworkX undirected graph.
        dependency_matrix (np.ndarray): Matrix for storing computed dependencies.
        x (int): Index of x node.
        y (int): Index of y node.

    Returns:
        float: Computed dependency of node X on node Y.
    """
    if dependency_matrix[x][y] != -1:
        return dependency_matrix[x][y]
    
    weight_xy = get_edge_weight(G, x, y)

    CN_xy = nx.common_neighbors(G, x, y)

    sum_CN = 0.0

    sum_CN += weight_xy
    for v_i in CN_xy:
        weight_xvi = get_edge_weight(G, x, v_i)

        r_x_vi_y = r(G, x, v_i, y)

        sum_CN += weight_xvi * r_x_vi_y

    N_x = G.neighbors(x)

    sum_N = 0.0

    for v_j in N_x:
        weight_xvj = get_edge_weight(G, x, v_j)

        sum_N += weight_xvj

    D_xy = sum_CN / sum_N
    dependency_matrix[x][y] = D_xy

    return D_xy
