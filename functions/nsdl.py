import random 
import numpy as np
import networkx as nx
import math


class DependencyEmbedding:
    """
    This is abstract class for dependency embedding.
    It is intended to use only with undirected graphs.

    These functions have to to be implemented in child class:
    - iteration - one iteration of genetic embedding algorithm
    - run - run genetic embedding algorithm for given number of iterations
    """


    def __init__(self,
                 network: nx.Graph,
                 dependency_matrix: np.ndarray,
                 embedding_dim: int,
                 embedding_scale: float = 1.0) -> None:    
        

        self.network = network
        self.dependency_matrix = dependency_matrix
        self.embedding_dim = embedding_dim
        self.embedding_scale = embedding_scale

        self.embedding_current_state = np.random.uniform(low=0.0,
                                                         high=self.embedding_scale,
                                                         size=(len(network.nodes), self.embedding_dim))
        self.embedding_next_state = self.embedding_current_state.copy()

    def iteration(self):
        "Virtual function - should be implemented in child class"
        pass

    def run(self,iterations):
        for _ in range(iterations):
            self.iteration()
            self.embedding_current_state = self.embedding_next_state.copy()

        # return embeddings
        return self.embedding_current_state
    

    def get_embeddings(self):
        return self.embedding_current_state

class DepFinal(DependencyEmbedding):

    def __init__( 
                self, network : nx.Graph,
                dependency_matrix : np.ndarray,
                embedding_dim : int,
                embedding_scale : float = 1.0,
                neighProb : float = 0.2,
                maxDist : float = 0.01,
                minDist : float = 0.0005 
                ):    
        super().__init__(network, dependency_matrix, embedding_dim, embedding_scale)

        self.neighProb = neighProb
        self.maxDist = maxDist * embedding_scale
        self.minDist = minDist * embedding_scale
        

    def choose_node(self,X):

        all_neighs = list(nx.neighbors(self.network,X))
        selected_neigh = random.choice(all_neighs)
        if random.random() < self.neighProb:
            return selected_neigh

        neighs_of_neigh = list(nx.neighbors(self.network, selected_neigh))
        neighs_of_neigh.remove(X)
        for node in all_neighs:
            if node in neighs_of_neigh:
                neighs_of_neigh.remove(node)

        if len(neighs_of_neigh) > 0:
            selected_neigh_of_neigh = random.choice(neighs_of_neigh)
            return selected_neigh_of_neigh

        while True:
            random_node = random.randint(0,len(self.network.nodes) - 1)
            if random_node != X:
                break
        return random_node 
        
    def iteration(self):
        
        for X in self.network.nodes:

            Y = self.choose_node(X)

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]
            norm_M = np.linalg.norm(M)
            M1 = M / norm_M    

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]

            q = D_x_y * D_y_x * (D_x_y + D_y_x) / 2.0

            expDist = (1 - q) * self.maxDist
            expMinDist = (1 - q) * self.minDist

            exp = norm_M / expDist
            exp = 1.0 / (0.5 + exp / 1.5)
            self.embedding_next_state[X] = self.embedding_current_state[X] + math.pow(q, exp) * (norm_M - expMinDist) * M1
