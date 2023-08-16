import networkx as nx
import numpy as np


def r(G,x,vi,y):
  
  weight_viy = get_edge_weight(G, vi, y)
  weight_xvi = get_edge_weight(G, x, vi)
  weight_viy = get_edge_weight(G, vi, y)

  return weight_viy / (weight_xvi + weight_viy)


def dependency(G,has_edge_weights = False):
  for node in G.nodes:
    for edge in G.edges(node):

      x,y = edge 

      weight_xy = 1.0
      if has_edge_weights:
        weight_xy = G[x][y]['weight']

      CN_xy = list(list(set(G.successors(x)) & set(G.predecessors(y))))

      sum_CN = 0.0

      sum_CN += weight_xy
      for v_i in CN_xy:

        weight_xvi = 1.0 
        if has_edge_weights:
          weight_xvi = G[x][y]['weight']

        r_x_vi_y = r(G,x,v_i,y,has_edge_weights)

        sum_CN += (weight_xvi * r_x_vi_y)

      N_x = G.neighbors(x)

      sum_N = 0.0

      for v_j in N_x:
        weight_xvj = 1.0
        if has_edge_weights:
          weight_xvj = G[x][v_j]['weight']

        sum_N += weight_xvj
       
      D_xy = sum_CN / sum_N
      G[x][y]['dep'] = D_xy

def get_edge_weight(G, x, y):
    if G.has_edge(x,y):
        return G[x][y]['weight']
    else:
        return 0.0
   
def dependency_matrix(G):

  dep_matrix = np.zeros((len(G.nodes),len(G.nodes)))

  for x in G.nodes:
    for y in G.nodes:

      if not G.has_edge(x,y): continue

      if x == y: continue

      weight_xy = get_edge_weight(G, x, y)

      CN_xy = nx.common_neighbors(G, x, y)

      sum_CN = 0.0

      sum_CN += weight_xy
      for v_i in CN_xy:
        weight_xvi = get_edge_weight(G, x, v_i)

        r_x_vi_y = r(G,x,v_i,y)

        sum_CN += (weight_xvi * r_x_vi_y)

      N_x = G.neighbors(x)

      sum_N = 0.0

      for v_j in N_x:
        weight_xvj = get_edge_weight(G, x, v_j)

        sum_N += weight_xvj
      
      D_xy = sum_CN / sum_N
      dep_matrix[x][y] = D_xy

  return dep_matrix
