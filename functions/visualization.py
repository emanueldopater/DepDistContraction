from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx
from functions.nsdl import DependencyEmbedding

# generate functions that save only one picture on the N iteration




def visualize_network_animation(
        G: nx.Graph,
        embedding_generator: DependencyEmbedding,
        iterations: int,
        visualization_step: int,
        scatter_size_offset: int = 1,
        scatter_size_degree_power: int = 1.9,
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
        scatter = ax.scatter(node_embeddings[:, 0], node_embeddings[:, 1], color='blue', s=[scatter_size_offset + G.degree[node] ** scatter_size_degree_power for node in sorted(G.nodes)])
        
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


def generate_images_and_gdfs_at_n(
        G: nx.Graph,
        embedding_generator: DependencyEmbedding,
        iterations: int,
        visualization_step: int,
        scatter_size_offset: int = 1,
        scatter_size_degree_power: int = 1.9,
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
        scatter = ax.scatter(node_embeddings[:, 0], node_embeddings[:, 1], color='blue', s=[scatter_size_offset + G.degree[node] ** scatter_size_degree_power for node in sorted(G.nodes)])
        
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