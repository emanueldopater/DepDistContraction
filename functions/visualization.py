from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx
from functions.depdist_contraction import DepDist_Base
from functions.exporter import export_to_gdf
# generate functions that save only one picture on the N iteration


def pdf_gdf_after_n_iterations(
        G: nx.Graph,
        embedding_generator: DepDist_Base,
        iterations: int,
        show_iterations : list[int],
        node_display_size_base: int = 1,
        node_display_size_power: int = 1.9,
        show_labels : bool = False,
        file_prefix : str = ""
        ) -> None:
    """
    Generates

    Args:
        G (nx.Graph): _description_
        embedding_generator (DepDist_Base): _description_
        iterations (int): _description_
        show_iterations (list[int]): _description_
        node_display_size_base (int, optional): _description_. Defaults to 1.
        node_display_size_power (int, optional): _description_. Defaults to 1.9.
        show_labels (bool, optional): _description_. Defaults to False.
        file_prefix (str, optional): _description_. Defaults to "".
    """

    for i in range(iterations):
        embs = embedding_generator.run(iterations=1)

        if (i+1) in show_iterations:
            # Draw the edges
            for edge in G.edges:
                src, tar = edge
                x = [embs[src][0], embs[tar][0]]
                y = [embs[src][1], embs[tar][1]]
                plt.plot(x, y, color='black', linewidth=0.1)


            # Update the scatter plot with the new node positions
            plt.scatter(embs[:, 0], embs[:, 1], color='blue', s=[node_display_size_base + G.degree[node] ** node_display_size_power for node in sorted(G.nodes)])
            
            if show_labels:
                for node in G.nodes:
                    node_emb = embs[node]
                    plt.annotate(str(node), (node_emb[0], node_emb[1]), color='red', fontsize=12)


            #show iteration number as title of plot
            plt.title("Iteration: " + str(i+1))

            #save plot to pdf
            plt.savefig(file_prefix + "_iteration_" + str(i+1) + ".pdf")
            export_to_gdf(file_prefix + "_iteration_" + str(i+1) + ".gdf",G,embs,has_labels=show_labels)
            plt.clf()






def visualize_network_animation(
        G: nx.Graph,
        embedding_generator: DepDist_Base,
        iterations: int,
        visualization_step: int,
        node_display_size_base: int = 1,
        node_display_size_power: int = 1.9,
         fps : int = 30,
         show_labels : bool = False,
         ):
    
    ### generate embeddings### 
    list_of_embs = []
    list_of_embs.append(embedding_generator.get_embeddings())
    for i in range(iterations):
        embs = embedding_generator.run(iterations=1)
        if i % visualization_step == 0:
            list_of_embs.append(embs) 

    ############################

    fig, ax = plt.subplots()

    # Create empty scatter and plot objects
    scatter = ax.scatter([], [])
    edges, = ax.plot([], [], color='black', linewidth=0.1)

    def update(iteration):
        ax.clear()

        # Get the node embeddings for the current iteration
        node_embeddings = list_of_embs[iteration]

        # Update the scatter plot with the new node positions
        scatter = ax.scatter(node_embeddings[:, 0], node_embeddings[:, 1], color='blue', s=[node_display_size_base + G.degree[node] ** node_display_size_power for node in sorted(G.nodes)])
        
        if show_labels:
            for node in G.nodes:
                node_emb = node_embeddings[node]
                ax.annotate(str(node), (node_emb[0], node_emb[1]), color='red', fontsize=12)

        # Draw the edges
        for edge in G.edges:
            src, tar = edge
            x = [node_embeddings[src][0], node_embeddings[tar][0]]
            y = [node_embeddings[src][1], node_embeddings[tar][1]]
            ax.plot(x, y, color='black', linewidth=0.1)


        #show iteration number as title
        ax.set_title("Iteration: " + str(iteration * visualization_step) + "/" + str(iterations))

        return scatter, edges

    # Create the animation
    num_iterations = len(list_of_embs)
    animation_fps = int(1000 / fps)
    animation = FuncAnimation(fig, update, frames=num_iterations, interval=animation_fps)
    
    plt.show()


    

def visualize_network_gif(G: nx.Graph, embs_list: list, output_path: str, scatter_size_offset: int = 2, scatter_size_degree_power: int = 2, fps : int = 30):
    fig, ax = plt.subplots()

    # Create empty scatter and plot objects
    scatter = ax.scatter([], [])
    edges, = ax.plot([], [], color='black', linewidth=0.1)

    def update(iteration):
        ax.clear()

        # Get the node embeddings for the current iteration
        node_embeddings = embs_list[iteration]

        # Update the scatter plot with the new node positions
        scatter = ax.scatter(node_embeddings[:, 0], node_embeddings[:, 1], color='blue', s=[scatter_size_offset + G.degree[node] ** scatter_size_degree_power for node in sorted(G.nodes)])
        for node in G.nodes:
            node_emb = node_embeddings[node]
            ax.annotate(str(node), (node_emb[0], node_emb[1]), color='red', fontsize=12)

        # Draw the edges
        for edge in G.edges:
            src, tar = edge
            x = [node_embeddings[src][0], node_embeddings[tar][0]]
            y = [node_embeddings[src][1], node_embeddings[tar][1]]
            ax.plot(x, y, color='black', linewidth=0.1)

        return scatter, edges

    # Create the animation
    num_iterations = len(embs_list)
    animation_fps = int(1000 / fps)
    animation = FuncAnimation(fig, update, frames=num_iterations, interval=animation_fps)

    # Show the animation
    animation.save(output_path, writer='ffmpeg')