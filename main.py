from functions.data_loading import *
from functions.dependency import *
from functions.depdist_contraction import *
from functions.visualization import *
from functions.exporter import *


# loading networks
loaded_karate_club = load_net_from_edge_list(path='datasets/karateclub_edges.csv', sep=';', has_edge_weights = False)
# loaded_lemis = load_net_from_edge_list(path="datasets/lesmis_edges.csv", sep=";", has_edge_weights=True)
loaded_net_science = load_net_from_edge_list(path='datasets/ca-netscience_edges.csv', sep=' ', has_edge_weights = False)
# loaded_football = load_net_from_edge_list(path='datasets/edges_football.csv', sep=';', has_edge_weights = False)


# creating embedding generators
embedding_karate_club = DepDist_Contraction(
    network=loaded_karate_club,
    embedding_dim=2
)

# embedding_lesmis = DepDist_Contraction(
#     network=loaded_lemis,
#     embedding_dim=2
# )

# embedding_football = DepDist_Contraction(
#     network=loaded_football,
#     embedding_dim=2
# )

embedding_net_science = DepDist_Contraction(
    network=loaded_net_science,
    embedding_dim=2
)


visualize_network_animation(
    G=loaded_net_science,
    embedding_generator=embedding_net_science,
    iterations=500,
    visualization_step=5,
    show_labels=False,
    fps=1
)




# pdf_gdf_after_n_iterations(
#     G=loaded_karate_club,
#     embedding_generator=embedding_karate_club,
#     iterations=500,
#     show_iterations=[50,500],
#     show_labels=False,
#     file_name="karate_club"
# )


# this function will run the embedding generator for 500 iterations within itself
# and will generate animation of the network.
# visualize_network_animation(
#     G=loaded_karate_club,
#     embedding_generator=embedding_karate_club,
#     iterations=500,
#     visualization_step=5,
#     show_labels=False,
# )

# visualize_network_animation(
#     G=loaded_lemis,
#     embedding_generator=embedding_lesmis,
#     iterations=500,
#     visualization_step=5,
#     show_labels=False,
# )


