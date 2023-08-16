from functions.data_loading import *
from functions.dependency import *
from functions.nsdl import *
from functions.visualization import *
from functions.exporter import *

#loaded_net = load_net_from_edge_list(path='datasets/karateclub_edges.csv', sep=';', has_edge_weights = False)
loaded_net = load_net_from_edge_list(path='datasets/lesmis_edges.csv', sep=';', has_edge_weights = True)
#loaded_net = load_net_from_edge_list(path='datasets/ca-netscience_edges.csv', sep=' ', has_edge_weights = False)
#loaded_net = load_net_from_edge_list(path='datasets/edges_football.csv', sep=';', has_edge_weights = False)


dep_matrix = dependency_matrix(loaded_net)



genetic_embedding = DepFinal( network=loaded_net,
                              dependency_matrix=dep_matrix,
                              embedding_dim=2,
                              minDist=0.0)

visualize_network_animation(G=loaded_net,
                            embedding_generator=genetic_embedding,
                            iterations=3000,
                            visualization_step=50,
                            show_labels=False)
