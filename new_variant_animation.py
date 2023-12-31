from functions.data_loading import *
from functions.dependency import *
from functions.genetic_embedding import *
from functions.visualization import *

karate_network = load_net_from_edge_list(path='datasets/karateclub_edges.csv', sep=';', has_edge_weights = False)

dep_matrix = dependency_matrix(karate_network)

genetic_embedding = GeneticEmbeddingNova(network=karate_network, dependency_matrix=dep_matrix, embedding_dim=2, max_step_portion=0.1)

iterations = 3000
list_of_embs = []
for i in range(iterations):
    embs = genetic_embedding.run(iterations=1)
    list_of_embs.append(embs)

visualize_network_animation(karate_network, list_of_embs,'animation_lesmis_2.gif', scatter_size_offset=3, scatter_size_degree_power=1.8)