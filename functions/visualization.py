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
        '''
        if node.label is not None:
        plt.annotate(str(node.label), (emb[0],emb[1]),color='red',fontsize=12)

        else:
        plt.annotate(str(int(node.id)), (emb[0],emb[1]),color='red',fontsize=12)

        '''
        plt.annotate(str(node), (node_emb[0],node_emb[1]),color='red',fontsize=12)

    plt.show()