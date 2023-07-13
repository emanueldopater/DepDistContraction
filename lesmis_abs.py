from functions.data_loading import *
from functions.dependency import *
from functions.genetic_embedding import *
from functions.visualization import *

lesmis_network = load_net_from_edge_list(path='datasets/lesmis_edges.csv', sep=';', has_edge_weights = True)

dep_matrix = dependency_matrix(lesmis_network)

genetic_embedding = DependencyEmebeddingDocument_v13_07_2023(network=lesmis_network, dependency_matrix=dep_matrix, embedding_dim=2)


#print("Max dist: ", math.log(len(lesmis_network.nodes)) * math.sqrt(len(lesmis_network.nodes)))

# genetic_embedding = GeneticEmbeddingAbsLog(network=lesmis_network, dependency_matrix=dep_matrix, embedding_dim=2,
#                                                 max_step_portion=0.5)

iterations = 5000
list_of_embs = []
list_of_embs.append(genetic_embedding.get_embeddings())
for i in range(iterations):
    embs = genetic_embedding.run(iterations=1)
    if i % 50 == 0:
        list_of_embs.append(embs)   

#visualize_network_animation(lesmis_network, list_of_embs,'animation_0_1.gif', scatter_size_offset=1, scatter_size_degree_power=1.9, fps=2)
visualize_network_gif(lesmis_network, list_of_embs,'animations/animation_13_7_new.gif', scatter_size_offset=1, scatter_size_degree_power=1.9, fps=2)