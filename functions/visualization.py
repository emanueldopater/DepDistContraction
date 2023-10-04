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
        file_prefix : str = "",
        n_step_to_save : int = 5
        ) -> None:
    """
    Function which generates pdf plots and gdf files after specified number of iterations.

    Args:
        G (nx.Graph): NetworkX undirected graph. 
        embedding_generator (DepDist_Base): Embedding generator.
        iterations (int): Number of iterations to run.
        show_iterations (list[int]): List of ints indicating which iterations to show.
        node_display_size_base (int, optional): This paramter specify the size of node in plot. Defaults to 1.
        node_display_size_power (float, optional): This paramter specify the size of node in plot. Defaults to 1.9.
        show_labels (bool, optional): Whether to draw labels to plot. Defaults to False.
        file_prefix (str, optional):File prexix. Defaults to "".
    """

    embs_to_return = []
    for i in range(iterations):
        embs = embedding_generator.run(iterations=1)

        if (i % n_step_to_save) == 0:
            embs_to_return.append(embs)
            
        if (i+1) in show_iterations:

            

            # Draw the edges
            for edge in G.edges:
                src, tar = edge
                x = [embs[src][0], embs[tar][0]]
                y = [embs[src][1], embs[tar][1]]
                plt.plot(x, y, color='black', linewidth=0.05, zorder=0)

            #draw nodes
            plt.scatter(embs[:, 0], 
                        embs[:, 1], 
                        color='blue', 
                        s=[node_display_size_base + G.degree[node] ** node_display_size_power for node in sorted(G.nodes)], 
                        zorder=5)


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
    return embs_to_return


def visualize_network_animation(
            G: nx.Graph,
            embedding_generator: DepDist_Base,
            iterations: int,
            visualization_step: int,
            node_display_size_base: int = 1,
            node_display_size_power: int = 1.9,
            fps : int = 30,
            show_labels : bool = False,
            ) -> None:
    
    """
    Create animation of layout.

    Args:
        G (nx.Graph): NetworkX undirected graph. 
        embedding_generator (DepDist_Base): Embedding generator.
        iterations (int): Number of iterations to run.
        node_display_size_base (int, optional): This paramter specify the size of node in plot. Defaults to 1.
        node_display_size_power (float, optional): This paramter specify the size of node in plot. Defaults to 1.9.
        fps (int, optional): Frames per second. Defaults to 30.
        show_labels (bool, optional): Whether to draw labels to plot. Defaults to False.
    """
    
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


    
def visualize_network_gif(G: nx.Graph,
                        embs_list : list,
                        node_display_size_base: int = 1,
                        node_display_size_power: int = 1.9,
                        file_prefix : str = "", 
                        visualization_step : int = 5,
                        iterations: int = 500,
                        fps : int = 30):
    fig, ax = plt.subplots()

    # Create empty scatter and plot objects
    scatter = ax.scatter([], [])
    edges, = ax.plot([], [], color='black', linewidth=0.1)

    def update(iteration):
        ax.clear()

        # Get the node embeddings for the current iteration
        node_embeddings = embs_list[iteration]

        # Update the scatter plot with the new node positions
        scatter = ax.scatter(node_embeddings[:, 0], node_embeddings[:, 1], color='blue', s=[node_display_size_base + G.degree[node] ** node_display_size_power for node in sorted(G.nodes)])
        for node in G.nodes:
            node_emb = node_embeddings[node]
            #ax.annotate(str(node), (node_emb[0], node_emb[1]), color='red', fontsize=12)

        # Draw the edges
        for edge in G.edges:
            src, tar = edge
            x = [node_embeddings[src][0], node_embeddings[tar][0]]
            y = [node_embeddings[src][1], node_embeddings[tar][1]]
            ax.plot(x, y, color='black', linewidth=0.1)


        ax.set_title("Iteration: " + str(iteration * visualization_step) + "/" + str(iterations))

        return scatter, edges

    # Create the animation
    num_iterations = len(embs_list)
    animation_fps = int(1000 / fps)
    animation = FuncAnimation(fig, update, frames=num_iterations, interval=animation_fps)

    # Show the animation
    animation.save(file_prefix , writer='imagemagick')