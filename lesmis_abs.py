from functions.data_loading import *
from functions.dependency import *
from functions.genetic_embedding import *
from functions.visualization import *
from functions.exporter import *

#lesmis_network = load_net_from_edge_list(path='datasets/karateclub_edges.csv', sep=';', has_edge_weights = False)
lesmis_network = load_net_from_edge_list(path='datasets/lesmis_edges.csv', sep=';', has_edge_weights = True)
#lesmis_network = load_net_from_edge_list(path='datasets/ca-netscience_edges.csv', sep=' ', has_edge_weights = False)
#lesmis_network = load_net_from_edge_list(path='datasets/edges_football.csv', sep=';', has_edge_weights = False)
#lesmis_network = load_net_from_edge_list(path='datasets/condmat_edges.csv', sep=';', has_edge_weights = True)


dep_matrix = dependency_matrix(lesmis_network)
print(lesmis_network.nodes)

print(lesmis_network.degree)

# genetic_embedding = DepFinal(network=lesmis_network,
#                                 dependency_matrix=dep_matrix,
#                                 embedding_dim=2)


# iterations = 5000
# list_of_embs = []
# list_of_embs.append(genetic_embedding.get_embeddings())
# for i in range(iterations):
#     embs = genetic_embedding.run(iterations=1)
#     if i % 50 == 0:
#         list_of_embs.append(embs)   

#export_to_gdf('gdf_files/fkarate_4_8.gdf' , lesmis_network, genetic_embedding.get_embeddings(), has_labels=True)
#visualize_network_animation(lesmis_network, list_of_embs,scatter_size_offset=1, scatter_size_degree_power=1.9, fps=15)
#visualize_network_gif(lesmis_network, list_of_embs,'animations/animation_new_version_27.gif', scatter_size_offset=1, scatter_size_degree_power=1.9, fps=10) 