from functions.data_loading import *
from functions.dependency import *
from functions.nsdl import *
from functions.visualization import *
from functions.exporter import *

loaded_net = load_net_from_edge_list(path='datasets/karateclub_edges.csv', sep=';', has_edge_weights = False)
# loaded_net = load_net_from_edge_list(
#     path="datasets/lesmis_edges.csv", sep=";", has_edge_weights=True
# )
# loaded_net = load_net_from_edge_list(path='datasets/ca-netscience_edges.csv', sep=' ', has_edge_weights = False)
# loaded_net = load_net_from_edge_list(path='datasets/edges_football.csv', sep=';', has_edge_weights = False)


dep_matrix = dependency_matrix(loaded_net)


genetic_embedding = DependencyEmebeddingDocument_DepDist(
    network=loaded_net,
    dependency_matrix=dep_matrix,
    embedding_dim=2
)

visualize_network_animation(
    G=loaded_net,
    embedding_generator=genetic_embedding,
    iterations=500,
    visualization_step=10,
    show_labels=True,
)
export_to_gdf(
    filename="output.gdf", G=loaded_net, embs=genetic_embedding.get_embeddings()
)
