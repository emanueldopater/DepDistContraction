import networkx as nx
import numpy as np


def r(G, x, vi, y):
    weight_viy = get_edge_weight(G, vi, y)
    weight_xvi = get_edge_weight(G, x, vi)

    return weight_viy / (weight_xvi + weight_viy)


def get_edge_weight(G, x, y):
    if G.has_edge(x, y):
        return G[x][y]["weight"]
    else:
        return 0.0


def dependency_matrix(G):
    dep_matrix = np.zeros((len(G.nodes), len(G.nodes)))

    for x in G.nodes:
        for y in G.nodes:
            if x == y:
                continue

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
            dep_matrix[x][y] = D_xy

    return dep_matrix


def dependency_X_Y(G, dependency_matrix, x,y ) -> float:
    #dep_matrix = np.zeros((len(G.nodes), len(G.nodes)))

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
