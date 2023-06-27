import random 
import numpy as np
import networkx as nx


class GeneticEmbedding:
    """
    This is abstract class for genetic embedding.
    It is intended to use only with undirected graphs.

    These functions have to to be implemented in child class:
    - iteration - one iteration of genetic embedding algorithm
    - run - run genetic embedding algorithm for given number of iterations
    - precompute_dependency_probabilities - precompute probabilities of choosing neighbour for each node.
        This probabilities will be always the same.
    """


    def __init__(self, network : nx.Graph, dependency_matrix : np.ndarray, embedding_dim : int):    
        self.network = network
        self.dependency_matrix = dependency_matrix
        self.embedding_dim = embedding_dim

        self.embedding_current_state = np.random.uniform(size=(len(network.nodes), self.embedding_dim))
        self.embedding_next_state = self.embedding_current_state.copy()

        self.precomputed_dep_probs = np.zeros((len(self.network.nodes), len(self.network.nodes)))

        self.precompute_dependency_probabilities()


    def iteration(self):
        "Virtual function - should be implemented in child class"
        pass

    def run(self, iterations : float):
        "Virtual function - should be implemented in child class"
        pass

    def precompute_dependency_probabilities(self):

        """
        This functions precomputes probabilities based on dependencies and normalizes it to sum to 1.0.
        It is worth to precompute it, because it will be always the same.

        """
        for node in self.network.nodes:
            sum_of_deps = 0.0

            for dep_value in self.dependency_matrix[node]:
                sum_of_deps += dep_value

            deps_scaled = self.dependency_matrix[node] / sum_of_deps
            cumulative_probs = [sum(deps_scaled[:i+1]) for i in range(len(deps_scaled))]

            self.precomputed_dep_probs[node] = cumulative_probs

    def get_embeddings(self):
        return self.embedding_current_state


class GeneticEmbeddingV1(GeneticEmbedding):
    """
    This is the first version of algorithm.

    Algorithm is performed in following way:

    For each node in network:

        Step 1: We decide on probabily (1 / (num_neigh + 1)) whether node will change it embedding or not 
        (the more neigbour node has, the less probability it has to change its embedding)

        Step 2: If node decides to change its embedding, it chooses one of its neighbours with probability proportional to dependency.
        Dependencies of its neighbours are normalized to sum to 1.0. We use cumulative distribution function to choose neighbour.

        Step 3 : We assign node embedding of chosen neighbour + random perturberation - normal distribution with mean 0 and always std 1.0.
        This assign it to node's next state embedding.
    """


    def __init__(self, network : nx.Graph, dependency_matrix : np.ndarray, embedding_dim : int):    
        super().__init__(network, dependency_matrix, embedding_dim)



    def run(self,iterations):

        for i in range(iterations):
            self.iteration()

            # update embedding
            self.embedding_current_state = self.embedding_next_state.copy()

        # return embeddings

        return self.embedding_current_state

    def iteration(self):
        for current_node in self.network.nodes:

            # Step1: Decide whether node will change its embedding or not
            num_neigh = self.network.degree[current_node]
            prob_of_change = (1 / (num_neigh + 1))

            if random.random() <  prob_of_change:

                # Step2: Choose neighbour to change embedding

                # get precomputed cumulative probability
                cumulative_probs = self.precomputed_dep_probs[current_node]

                random_number = random.random()
                chosen_neigh = None

                for i, cumulative_prob in enumerate(cumulative_probs):
                    if random_number <= cumulative_prob:
                        chosen_neigh = i
                        break
                
                # Step3: Change embedding

                tar_emb = self.embedding_current_state[chosen_neigh]

                perturberation = np.random.normal(scale=1.0, size=self.embedding_dim)

                self.embedding_next_state[current_node] = tar_emb + perturberation


class GeneticEmbeddingV2(GeneticEmbedding):
    """
    This is the second version of algorithm. It is the same as first version,
    but here we are gradually decreasing the standard deviation of perturberation towards 0.0. 
    Standard deviation starts at 1.0 and each iteration it degreeses by value 1 / iterations (so at the end it reaches 0).

    """

    def __init__(self, network : nx.Graph, dependency_matrix : np.ndarray, embedding_dim : int):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.perturberation_std = 1.0


    def run(self,iterations):

        std_decrease_rate = self.perturberation_std / iterations

        for i in range(iterations):
            self.iteration()

            # update embedding
            self.embedding_current_state = self.embedding_next_state.copy()

            # decrease std of perturberation
            self.perturberation_std -= std_decrease_rate

        # return embeddings

        return self.embedding_current_state

    def iteration(self):
        for current_node in self.network.nodes:

            # Step1: Decide whether node will change its embedding or not
            num_neigh = self.network.degree[current_node]
            prob_of_change = (1 / (num_neigh + 1))

            if random.random() <  prob_of_change:

                # Step2: Choose neighbour to change embedding

                # get precomputed cumulative probability
                cumulative_probs = self.precomputed_dep_probs[current_node]

                random_number = random.random()
                chosen_neigh = None

                for i, cumulative_prob in enumerate(cumulative_probs):
                    if random_number <= cumulative_prob:
                        chosen_neigh = i
                        break
                
                # Step3: Change embedding 
                # Instead 1.0 as std of perturberation we use self.perturberation_std

                tar_emb = self.embedding_current_state[chosen_neigh]

                perturberation = np.random.normal(scale=self.perturberation_std, size=self.embedding_dim)

                self.embedding_next_state[current_node] = tar_emb + perturberation


class GeneticEmbeddingV3(GeneticEmbedding):
    """
    This is the third version of algorithm. It is similar to the first 2 version with first 2 steps, but the third step is different:
    Step 3: The node will move towards the chosen neighbour the following way:
        - parameter max_step_portion determines maximum portion of possible step towards the chosen neighbour. So the value 0.5 means that 
        node can be moved maximally by half (or 50%) of the distance towards the chosen neighbour. 
        - The  step is determined by the dependency on the chosen neighbour. The higher dependency, the bigger step towards the chosen neighbour.

        So the amount of step towards the chosen neighbour is determined by max_step_portion * dependency. 
        Simply:
            step = max_step_portion * dependency

            new_embedding += (neigh_emb - current_node_emb) * step


    """

    def __init__(self, network : nx.Graph, dependency_matrix : np.ndarray, embedding_dim : int, max_step_portion : float = 0.5):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.max_step_portion = max_step_portion


    def run(self,iterations):
        for i in range(iterations):
            self.iteration()

            # update embedding
            self.embedding_current_state = self.embedding_next_state.copy()

        # return embeddings
        return self.embedding_current_state

    def iteration(self):
        for current_node in self.network.nodes:

            # Step1: Decide whether node will change its embedding or not
            num_neigh = self.network.degree[current_node]
            prob_of_change = (1 / (num_neigh + 1))

            #if random.random() <  prob_of_change:

            # Step2: Choose neighbour to change embedding

            # get precomputed cumulative probability
            cumulative_probs = self.precomputed_dep_probs[current_node]

            random_number = random.random()
            chosen_neigh = None

            for i, cumulative_prob in enumerate(cumulative_probs):
                if random_number <= cumulative_prob:
                    chosen_neigh = i
                    break


            dependency_on_neigh = self.dependency_matrix[current_node][chosen_neigh]

            step = self.max_step_portion * dependency_on_neigh
            neigh_emb = self.embedding_current_state[chosen_neigh]
            current_node_emb = self.embedding_current_state[current_node]

            self.embedding_next_state[current_node] += (neigh_emb - current_node_emb) * step

class GeneticEmbeddingCenter(GeneticEmbedding):
    """
    This version is inspired by "center of gravity". It is similar to the third version, but instead of moving towards
    the one randomly chosen neighbour, we move towards the center of gravity of all neighbours. The "strength" of each node is its
    dependency.


    """

    def __init__(self, network : nx.Graph, dependency_matrix : np.ndarray, embedding_dim : int):    
        super().__init__(network, dependency_matrix, embedding_dim)



    def run(self,iterations):
        for i in range(iterations):
            self.iteration()

            # update embedding
            self.embedding_current_state = self.embedding_next_state.copy()

        # return embeddings
        return self.embedding_current_state

    def iteration(self):
        for current_node in self.network.nodes:

            # calculate center of gravity
            total_direction = 0.0
            current_node_possition = self.embedding_current_state[current_node]

            for neigh_node in self.network.nodes():

                dependency_on_neigh = self.dependency_matrix[current_node][neigh_node]
                neigh_position = self.embedding_current_state[neigh_node]

                direction = neigh_position - current_node_possition
                total_direction += direction * dependency_on_neigh
            

            self.embedding_next_state[current_node] += total_direction