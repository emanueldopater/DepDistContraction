import random
import numpy as np
import networkx as nx
import math
from functions.dependency import dependency_X_Y


class DependencyEmbedding:
    """
    This is abstract class for dependency embedding.
    It is intended to use only with undirected graphs.

    These functions have to to be implemented in child class:
    - iteration - one iteration of genetic embedding algorithm
    - run - run genetic embedding algorithm for given number of iterations
    """

    def __init__(
        self,
        network: nx.Graph,
        embedding_dim: int,
        embedding_scale: float = 1.0,
    ) -> None:
        self.network = network
        self.dependency_matrix = np.full((len(network.nodes), len(network.nodes)), -1.0)
        self.embedding_dim = embedding_dim
        self.embedding_scale = embedding_scale

        self.embedding_current_state = np.random.uniform(
            low=0.0,
            high=self.embedding_scale,
            size=(len(network.nodes), self.embedding_dim),
        )
        self.embedding_next_state = self.embedding_current_state.copy()



    def iteration(self):
        "Virtual function - should be implemented in child class"
        pass

    def run(self, iterations):
        for i in range(iterations):
            self.iteration()
            self.embedding_current_state = self.embedding_next_state.copy()

            print('Iteration: ' + str(i) + ' done')

        # return embeddings
        return self.embedding_current_state

    def get_embeddings(self):
        return self.embedding_current_state


class DependencyEmebeddingDocument_DepDist(DependencyEmbedding):
    def __init__(
        self, network: nx.Graph, embedding_dim: int
    ):
        super().__init__(network, embedding_dim)

        ####################################
        self.maxDist = 0.01
        self.minDist = 0.002
        ####################################

    def choose_node(self, X):
        all_neighs = list(nx.neighbors(self.network, X))

        neighProb = 1.0 - 1.0 / (1 + len(all_neighs))
        rand = random.random()

        selected_neigh = random.choice(all_neighs)
        if rand < neighProb:
            return selected_neigh

        neighs_of_neigh = list(nx.neighbors(self.network, selected_neigh))
        neighs_of_neigh.remove(X)
        for node in all_neighs:
            if node in neighs_of_neigh:
                neighs_of_neigh.remove(node)

        if len(neighs_of_neigh) > 0:
            selected_neigh_of_neigh = random.choice(neighs_of_neigh)
            return selected_neigh_of_neigh
        else:
            return selected_neigh

    def iteration(self):
        for X in self.network.nodes:
            Y = self.choose_node(X)

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]
            norm_M = np.linalg.norm(M)
            M1 = M / norm_M

            D_x_y = dependency_X_Y(self.network,self.dependency_matrix,X,Y)

            D_y_x = dependency_X_Y(self.network,self.dependency_matrix,Y,X)

            q = D_x_y * D_y_x * (D_x_y + D_y_x) / 2.0

            expDist = (1 - q) * self.maxDist
            expMinDist = (1 - q) * self.minDist

            accSpeed = 0.5 + 0.5 * norm_M / expDist

            self.embedding_next_state[X] = (
                self.embedding_current_state[X]
                + math.pow(q, 1.0 / accSpeed) * (norm_M - expMinDist) * M1
            )
