from functions.data_loading import *
from functions.dependency import *
from functions.depdist_contraction import *
from functions.visualization import *
from functions.exporter import *


# loading networks
loaded_mouse_brain = load_net_from_edge_list(path='datasets/mouse_brain_reindexed_0.csv', sep=';', has_edge_weights = False)
loaded_emails = load_net_from_edge_list(path="datasets/emails_reindexed_0.csv", sep=";", has_edge_weights=False)
loaded_airports = load_net_from_edge_list(path='datasets/airports_reindexed_0.csv', sep=';', has_edge_weights = False)


# creating embedding generators for each network
embedding_mouse_brain = DepDist_Contraction(
    network=loaded_mouse_brain,
    embedding_dim=2
)

embedding_emails = DepDist_Contraction(
    network=loaded_emails,
    embedding_dim=2
)

embedding_airports = DepDist_Contraction(
    network=loaded_airports,
    embedding_dim=2
)




# you can visualize the animation of embbedding generation with this function, it runs 500 iterations and diplay each 5th iteration

# here we visualize les miserales network
visualize_network_animation(
    G=loaded_airports,
    embedding_generator=embedding_airports,
    iterations=500,
    visualization_step=5,
    show_labels=False,
    fps=30,
    node_display_size_base=1,
    node_display_size_power=1.3
)


# or you can you use this function, which generates pdf images and gdf files for iteration specified in show_iterations list
# here we generate pdf and gdf for karate club network
# When opening gdf files, you need to use expansion layout, because node embeddings are in very small range (between 0 and 1).
# pdf_gdf_after_n_iterations(
#     G=loaded_karate_club,
#     embedding_generator=embedding_karate_club,
#     iterations=500,
#     show_iterations=[50,500],
#     show_labels=False,
#     file_prefix="karate_club"
# )

