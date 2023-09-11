from functions.data_loading import *
from functions.dependency import *
from functions.depdist_contraction import *
from functions.visualization import *
from functions.exporter import *

loaded_karate_club = load_net_from_edge_list(path='datasets/karateclub_edges.csv', sep=';', has_edge_weights = False)
loaded_lemis = load_net_from_edge_list(path="datasets/lesmis_edges.csv", sep=";", has_edge_weights=True)
loaded_net_science = load_net_from_edge_list(path='datasets/ca-netscience_edges.csv', sep=' ', has_edge_weights = False)
loaded_football = load_net_from_edge_list(path='datasets/edges_football.csv', sep=';', has_edge_weights = False)
#loaded_net = load_net_from_edge_list(path="datasets/condmat_edges.csv", sep=";", has_edge_weights=True)




embedding_net_science = DepDist_Contraction(
    network=loaded_net_science,
    embedding_dim=2
)

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

embedding_karate_club.run(500)
embedding_net_science.run(500)
embedding_lesmis.run(500)
embedding_football.run(500)


# visualize_network_animation(
#     G=loaded_net_science,
#     embedding_generator=embedding_net_science,
#     iterations=500,
#     visualization_step=5,
#     show_labels=False,
# )


# genetic_embedding_karate = DependencyEmebeddingDocument_DepDist(
#     network=loaded_karate_club,
#     embedding_dim=2
# )


# genetic_embedding_net_science.run(500)
# genetic_embedding_karate.run(500)




#logarithmic scale of y axis

#linewidth 0.5 

plt.plot(embedding_karate_club.differences,linewidth=0.8)
plt.plot(embedding_lesmis.differences, linewidth=0.8)
plt.plot(embedding_football.differences, linewidth=0.8 )
plt.plot(embedding_net_science.differences, linewidth=0.8)

# add legend
plt.legend(['karate', 'lesmis', 'football', 'netscience'])

# y label as mse
plt.ylabel('MSE')
plt.xlabel('Iteration')

plt.yscale("log")

# save plot as pdf
plt.savefig('mse.pdf')

plt.show()
# # export_to_gdf(
# #     filename="output.gdf", G=loaded_net, embs=genetic_embedding.get_embeddings()
# # )
