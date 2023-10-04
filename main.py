from functions.data_loading import *
from functions.dependency import *
from functions.depdist_contraction import *
from functions.visualization import *
from functions.exporter import *


# loading networks
loaded_karate_club = load_net_from_edge_list(path='datasets/karateclub_edges.csv', sep=';', has_edge_weights = False)
loaded_lemis = load_net_from_edge_list(path="datasets/lesmis_edges.csv", sep=";", has_edge_weights=True)
loaded_net_science = load_net_from_edge_list(path='datasets/ca-netscience_edges.csv', sep=' ', has_edge_weights = False)
loaded_football = load_net_from_edge_list(path='datasets/edges_football.csv', sep=';', has_edge_weights = False)


# creating embedding generators for each network



for i in range(3):


    embedding_karate_club = DepDist_Contraction(
    network=loaded_karate_club,
    embedding_dim=2
    )

    embedding_lesmis = DepDist_Contraction(
        network=loaded_lemis,
        embedding_dim=2
    )

    embedding_football = DepDist_Contraction(
        network=loaded_football,
        embedding_dim=2
    )

    embedding_net_science = DepDist_Contraction(
        network=loaded_net_science,
        embedding_dim=2
    )

    pdf_gdf_after_n_iterations(
        G=loaded_karate_club,
        embedding_generator=embedding_karate_club,
        iterations=500,
        show_iterations=[1],
        show_labels=False,
        file_prefix="data/karate_club_" + str(i),
        node_display_size_base=0.7,
        node_display_size_power=1.5
    )

    # pdf_gdf_after_n_iterations(
    #     G=loaded_lemis,
    #     embedding_generator=embedding_lesmis,
    #     iterations=500,
    #     show_iterations=[50,500],
    #     show_labels=False,
    #     file_prefix="data/lesmis_" + str(i),
    #     node_display_size_base=0.5,
    #     node_display_size_power=1.3
    # )

    # pdf_gdf_after_n_iterations(
    #     G=loaded_football,
    #     embedding_generator=embedding_football,
    #     iterations=500,
    #     show_iterations=[50,500],
    #     show_labels=False,
    #     file_prefix="data/football_" + str(i),
    #     node_display_size_base=0.5,
    #     node_display_size_power=1.3
    # )

    pdf_gdf_after_n_iterations(
        G=loaded_net_science,
        embedding_generator=embedding_net_science,
        iterations=500,
        show_iterations=[0],
        show_labels=False,
        file_prefix="data/net_science_" + str(i),
        node_display_size_base=0.2,
        node_display_size_power=1.1
    )


# you can visualize the animation of embbedding generation with this function, it runs 500 iterations and diplay each 5th iteration

# here we visualize les miserales network
# visualize_network_animation_3d(
#     G=loaded_lemis,
#     embedding_generator=embedding_lesmis,
#     iterations=500,
#     visualization_step=5,
#     show_labels=False,
#     fps=30
# )

# html_3d_after_n_iterations(
#     G=loaded_karate_club,
#     embedding_generator=embedding_karate_club,
#     iterations=500,
#     show_iterations=[500],
#     show_labels=False,
#     file_prefix="lesmis"

# )


# # or you can you use this function, which generates pdf images and gdf files for iteration specified in show_iterations list
# # here we generate pdf and gdf for karate club network
# # When opening gdf files, you need to use expansion layout, because node embeddings are in very small range (between 0 and 1). 
# pdf_gdf_after_n_iterations(
#     G=loaded_karate_club,
#     embedding_generator=embedding_karate_club,
#     iterations=500,
#     show_iterations=[50,500],
#     show_labels=False,
#     file_prefix="karate_club"
# )

