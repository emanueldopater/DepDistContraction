import random
import numpy as np
import networkx as nx
import math
from functions.dependency import dependency_X_Y
from sklearn.metrics import mean_squared_error

class DepDist_Base:
    """
    This is abstract class for dependency embedding.
    It is intended to use only with undirected graphs.

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

        # self.embedding_current_state = np.random.uniform(
        #     low=0.0,
        #     high=self.embedding_scale,
        #     size=(len(network.nodes), self.embedding_dim),
        # )

        self.embedding_current_state = np.zeros((len(network.nodes), self.embedding_dim))

        for node in sorted(network.nodes):
            portion = (1 - 1 / (1 + network.degree[node])) / 2
            self.embedding_current_state[node] = np.random.uniform(
                low=0.5 - portion,
                high=0.5 + portion,
                size=(self.embedding_dim,),
            )
   
        

        self.embedding_next_state = self.embedding_current_state.copy()


        self.differences = []


    def iteration(self):
        "Virtual function - should be implemented in child class"
        pass

    def run(self, iterations):
        for i in range(iterations):
            self.iteration()

            self.differences.append(mean_squared_error(self.embedding_current_state, self.embedding_next_state))
            
            self.embedding_current_state = self.embedding_next_state.copy()

            #print('Iteration: ' + str(i) + ' done')

        # return embeddings
        return self.embedding_current_state


    # obtain embeddings 
    def get_embeddings(self):
        return self.embedding_current_state



class DepDist_Contraction(DepDist_Base):
    def __init__( self, network : nx.Graph,
                  embedding_dim : int,
                ):    
        super().__init__(network, embedding_dim)

        ####################################
        self.maxDepDist = 0.002
        self.maxAccDist = 0.01
        ####################################

    def choose_node(self,X):

        all_neighs = list(nx.neighbors(self.network,X))

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

            depDist = (1.0 - q) * self.maxDepDist

            accDist = (1.0 - q) * self.maxAccDist
            accCoef = 0.5 + 0.5 * norm_M / accDist

            self.embedding_next_state[X] = self.embedding_current_state[X] + math.pow(q, 1.0 / accCoef) * (norm_M - depDist) * M1