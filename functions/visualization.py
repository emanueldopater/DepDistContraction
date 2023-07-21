from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_network(G :nx.Graph, embs : np.ndarray):

    #edges
    for edge in G.edges:
        src, tar = edge

        x = [embs[src][0], embs[tar][0]]
        y = [embs[src][1], embs[tar][1]]

        plt.plot(x,y,color='black',linewidth=0.1)

    #nodes
    for node in G.nodes:
        node_emb = embs[node]
        plt.scatter(node_emb[0], node_emb[1],color ='blue', s = [2 + G.degree[node]**2])
        plt.annotate(str(node), (node_emb[0],node_emb[1]),color='red',fontsize=12)

    plt.show()


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_network_animation(G: nx.Graph, embs_list: list, scatter_size_offset: int = 2, scatter_size_degree_power: int = 2, fps : int = 30):
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

        #ax.set_xlim(0, 1)
        #ax.set_ylim(0, 1)

        return scatter, edges

    # Create the animation
    num_iterations = len(embs_list)
    animation_fps = int(1000 / fps)
    animation = FuncAnimation(fig, update, frames=num_iterations, interval=animation_fps)

    # Show the animation
    #animation.save(output_path, writer='pillow')


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
