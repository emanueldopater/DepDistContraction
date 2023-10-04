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


# # creating embedding generators for each network
# embedding_karate_club = DepDist_Contraction(
#     network=loaded_karate_club,
#     embedding_dim=2
# )

# embedding_lesmis = DepDist_Contraction(
#     network=loaded_lemis,
#     embedding_dim=2
# )

# embedding_football = DepDist_Contraction(
#     network=loaded_football,
#     embedding_dim=2
# )

# embedding_net_science = DepDist_Contraction(
#     network=loaded_net_science,
#     embedding_dim=2
# )


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

    # karate 
    karate_embs = pdf_gdf_after_n_iterations(
        G=loaded_karate_club,
        embedding_generator=embedding_karate_club,
        iterations=500,
        show_iterations=[50,500],
        show_labels=False,
        file_prefix="data/karate_club_" + str(i),
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        n_step_to_save=5
    )

    visualize_network_gif(
        G=loaded_karate_club,
        embs_list=karate_embs,
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        file_prefix="data/karate_gif_" + str(i)+ ".gif",
        iterations=500,
        visualization_step=5,
        fps=5
    )

    # lesmis
    lesmis_embs = pdf_gdf_after_n_iterations(
        G=loaded_lemis,
        embedding_generator=embedding_lesmis,
        iterations=500,
        show_iterations=[50,500],
        show_labels=False,
        file_prefix="data/lesmis_" + str(i),
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        n_step_to_save=5
    )

    visualize_network_gif(
        G=loaded_lemis,
        embs_list=lesmis_embs,
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        file_prefix="data/lesmis_gif_" + str(i)+ ".gif",
        iterations=500,
        visualization_step=5,
        fps=5
    )

    # football
    football_embs = pdf_gdf_after_n_iterations(
        G=loaded_football,
        embedding_generator=embedding_football,
        iterations=500,
        show_iterations=[50,500],
        show_labels=False,
        file_prefix="data/football_" + str(i),
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        n_step_to_save=5
    )

    visualize_network_gif(
        G=loaded_football,
        embs_list=football_embs,
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        file_prefix="data/football_gif_" + str(i)+ ".gif",
        iterations=500,
        visualization_step=5,
        fps=5
    )

    # netscience
    netscience_embs = pdf_gdf_after_n_iterations(
        G=loaded_net_science,
        embedding_generator=embedding_net_science,
        iterations=500,
        show_iterations=[50,500],
        show_labels=False,
        file_prefix="data/netscience_" + str(i),
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        n_step_to_save=5
    )

    visualize_network_gif(
        G=loaded_net_science,
        embs_list=netscience_embs,
        node_display_size_base=0.7,
        node_display_size_power=1.5,
        file_prefix="data/netscience_gif_" + str(i)+ ".gif",
        iterations=500,
        visualization_step=5,
        fps=5
    )



    
