import random 
import numpy as np
import networkx as nx
import math


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

        self.embedding_current_state = np.random.uniform(low=-100, high=100, size=(len(network.nodes), self.embedding_dim))
        self.embedding_next_state = self.embedding_current_state.copy()

        self.precomputed_dep_probs = np.zeros((len(self.network.nodes), len(self.network.nodes)))

        self.precompute_dependency_probabilities()


    def iteration(self):
        "Virtual function - should be implemented in child class"
        pass

    def run(self,iterations):
        for _ in range(iterations):
            self.iteration()
            self.embedding_current_state = self.embedding_next_state.copy()

        # return embeddings
        return self.embedding_current_state

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

class GeneticEmbeddingNova(GeneticEmbedding):
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

            dependency_reverse = self.dependency_matrix[chosen_neigh][current_node]

            direction :float = 1.0

            if dependency_on_neigh < dependency_reverse:
                direction = -1.0
            

            step = self.max_step_portion * dependency_on_neigh


            neigh_emb = self.embedding_current_state[chosen_neigh]
            current_node_emb = self.embedding_current_state[current_node]

            self.embedding_next_state[current_node] += (neigh_emb - current_node_emb) * step * direction

class GeneticEmbeddingDifference(GeneticEmbedding):
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

            dependency_reverse = self.dependency_matrix[chosen_neigh][current_node]

            abs_diff = abs(dependency_on_neigh - dependency_reverse)

            direction :float = 1.0

            if dependency_on_neigh < dependency_reverse:
                direction = -1.0
            

            step = self.max_step_portion * abs_diff * dependency_on_neigh


            neigh_emb = self.embedding_current_state[chosen_neigh]
            current_node_emb = self.embedding_current_state[current_node]

            self.embedding_next_state[current_node] += (neigh_emb - current_node_emb) * step * direction

class GeneticEmbeddingV3Abs(GeneticEmbedding):
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

            dependency_reverse = self.dependency_matrix[chosen_neigh][current_node]

            abs_diff = abs(dependency_on_neigh - dependency_reverse)

            step = self.max_step_portion * dependency_on_neigh * ((abs_diff+1)/2)
            neigh_emb = self.embedding_current_state[chosen_neigh]
            current_node_emb = self.embedding_current_state[current_node]
            

            self.embedding_next_state[current_node] += (neigh_emb - current_node_emb) * step

class GeneticEmbeddingAbsLog(GeneticEmbedding):
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
            #prob_of_change = (1 / (num_neigh + 1))

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

            dependency_reverse = self.dependency_matrix[chosen_neigh][current_node]

            abs_diff = abs(dependency_on_neigh - dependency_reverse)

            step = self.max_step_portion * dependency_on_neigh * ((abs_diff+1)/2)
            
            neigh_emb = self.embedding_current_state[chosen_neigh]
            current_node_emb = self.embedding_current_state[current_node]
            dir_vector = (neigh_emb - current_node_emb)
            dist_of_nodes = math.dist(current_node_emb,neigh_emb)

            repulsion = np.arctan(10* dist_of_nodes - 0.01)

            self.embedding_next_state[current_node] += dir_vector * repulsion * step 

class GeneticEmbeddingThroughPairs(GeneticEmbedding):
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
        
        for n1 in self.network.nodes:
            for n2 in self.network.nodes:

                if n1 == n2: continue

                
                current_node = n1
                chosen_neigh = n2
                if random.random() < 0.5:
                    current_node = n2
                    chosen_neigh = n1
                


                dependency_on_neigh = self.dependency_matrix[current_node][chosen_neigh]

                dependency_reverse = self.dependency_matrix[chosen_neigh][current_node]

                abs_diff = abs(dependency_on_neigh - dependency_reverse)

                step = self.max_step_portion * dependency_on_neigh * ((abs_diff+1)/2)
                neigh_emb = self.embedding_current_state[chosen_neigh]
                current_node_emb = self.embedding_current_state[current_node]

                dir_vector = (neigh_emb - current_node_emb)

                repulsion = np.arctan(10* math.dist(current_node_emb,neigh_emb) - 0.01)

                #self.embedding_next_state[current_node] += dir_vector * repulsion * step
                self.embedding_next_state[current_node] += dir_vector * step

class GeneticEmbedding3Cases(GeneticEmbedding):
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
            dependency_reverse = self.dependency_matrix[chosen_neigh][current_node]

            direction = 1.0

            if dependency_on_neigh + dependency_reverse < 1.0:
                direction = -1.0

            


            gap = abs(1 - (dependency_on_neigh + dependency_reverse))


            #step = self.max_step_portion * dependency_on_neigh * ((abs_diff+1)/2)


            neigh_emb = self.embedding_current_state[chosen_neigh]
            current_node_emb = self.embedding_current_state[current_node]



            dir_vector_current_node = neigh_emb - current_node_emb
            dir_vector_neigh_node = current_node_emb - neigh_emb

            repulsion = np.arctan(10* math.dist(current_node_emb,neigh_emb) - 0.01)



            current_node_change = dir_vector_current_node * direction * repulsion * (1 - dependency_on_neigh) * gap
            neigh_node_change = dir_vector_neigh_node * direction * repulsion * (1 - dependency_reverse) * gap
            np.clip(current_node_change, None, 10 , out=current_node_change)
            np.clip(neigh_node_change, None, 10 , out=neigh_node_change)

            self.embedding_next_state[current_node] += current_node_change
            self.embedding_next_state[chosen_neigh] += neigh_node_change

class DependencyEmebeddingDocument(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray, embedding_dim : int,
                  MinDist : float,
                  MaxDist : float,
                  MinD : float = 0.1,
                  MaxD : float = 0.9,
                  MinDelta : float = 0.1
                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.MinDist = MinDist
        self.MaxDist = MaxDist
        self.MinD = MinD
        self.MaxD = MaxD
        self.MinDelta = MinDelta

    def run(self,iterations):


        for i in range(iterations):
            self.iteration()

            # update embedding
            self.embedding_current_state = self.embedding_next_state.copy()

            #print(self.embedding_current_state)

        # return embeddings
        return self.embedding_current_state

    def iteration(self):
        
        for X in self.network.nodes:
            
            Y = -1
            while True:

                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break

            
            if self.dependency_matrix[X][Y] == 0.0: continue
            
            #print("X,Y:", X,Y)
            D_x_y = max( min(self.dependency_matrix[X][Y], self.MaxD) , self.MinD)
            D_y_x = max( min(self.dependency_matrix[Y][X], self.MaxD) , self.MinD)

            M = self.embedding_current_state[X] - self.embedding_current_state[Y]
            #print("M: ", M)
            norm_M = np.linalg.norm(M)


            M1 = M / norm_M

            #print(M)
            
            
            #oddialenie
            if (D_x_y + D_y_x) < 1:
                q = (1 - D_x_y) * max(1 - D_x_y - D_y_x, self.MinDelta)* 0.1

                if norm_M == self.MaxDist:
                    self.embedding_next_state[X] = self.embedding_current_state[X]

                elif norm_M > self.MaxDist:
                    #self.embedding_next_state[X] = self.embedding_current_state[Y] + (self.MaxDist - self.MinDist) * M1
                    self.embedding_next_state[X] = self.embedding_current_state[Y] + (self.MaxDist) * M1

                else: # norm_M < self.MaxDist
                    self.embedding_next_state[X] = self.embedding_current_state[Y] + (q * (self.MaxDist - norm_M)) * M1
            
            #  D_x_y + D_y_x >= 1:
            else:
                q = D_x_y * max(abs(D_x_y - D_y_x), self.MinDelta) * 0.1

                if norm_M == self.MinDist:
                    self.embedding_next_state[X] = self.embedding_current_state[X]
                
                elif norm_M < self.MinDist:
                    #self.embedding_next_state[X] = self.embedding_current_state[Y] + 2 * self.MinDist * M1
                    self.embedding_next_state[X] = self.embedding_current_state[Y] + self.MinDist * M1

                #norm_M < self.MinDist
                else:
                    self.embedding_next_state[X] = self.embedding_current_state[X] - q * (norm_M - self.MinDist) * M1
            
class DependencyEmebeddingDocument_v13_07_2023(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray, embedding_dim : int,
                  MinWeakDist : float = 3.0,
                  MinStrongDist : float = 1.0,
                  MinD : float = 0.1,
                  MaxD : float = 0.9,
                  MinDelta : float = 0.1
                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.MinWeakDist = MinWeakDist
        self.MinStrongDist = MinStrongDist
        self.MinD = MinD
        self.MaxD = MaxD
        self.MinDelta = MinDelta
        

    def iteration(self):
        
        for X in self.network.nodes:
            
            Y = -1
            while True:

                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break

            
            #if self.dependency_matrix[X][Y] == 0.0: continue
            
            D_x_y = max( min(self.dependency_matrix[X][Y], self.MaxD) , self.MinD)
            D_y_x = max( min(self.dependency_matrix[Y][X], self.MaxD) , self.MinD)

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]

            #norm_M = np.linalg.norm(M)
            norm_M = math.sqrt(M[0]**2 + M[1]**2)
            M1 = M / norm_M            

            q = D_x_y * max((D_x_y + D_y_x) / 2, self.MinDelta) 



            # Situace1 (weak Dependency)
            if (D_x_y + D_y_x) < 1:
                

                if norm_M == self.MinWeakDist:
                    self.embedding_next_state[X] = self.embedding_current_state[X]

                elif norm_M < self.MinWeakDist:
                    self.embedding_next_state[X] = self.embedding_current_state[Y] - self.MinWeakDist * M1

                else: # norm_M > self.MinWeakDist
                    self.embedding_next_state[X] = self.embedding_current_state[X] + ( q * (norm_M - self.MinWeakDist)) * M1
            
            #  D_x_y + D_y_x >= 1:
            else:

                if norm_M == self.MinStrongDist:
                    self.embedding_next_state[X] = self.embedding_current_state[X]
                
                elif norm_M < self.MinStrongDist:
                    self.embedding_next_state[X] = self.embedding_current_state[Y] - self.MinStrongDist * M1

                #norm_M > self.MinStrongDist
                else:
                    self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - self.MinStrongDist) * M1       

class DependencyEmebeddingDocument_v23_07_2023(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  MinDist = 1.0
                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.MinDist = MinDist
        

    def iteration(self):
        
        for X in self.network.nodes:
            
            Y = -1
            while True:

                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break

            
            #if self.dependency_matrix[X][Y] == 0.0: continue
            
            M = self.embedding_current_state[Y] - self.embedding_current_state[X]

            norm_M = np.linalg.norm(M)
            M1 = M / norm_M            

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]
            q = D_x_y * ((D_x_y + D_y_x) / 2)
                

            if norm_M == self.MinDist:
                self.embedding_next_state[X] = self.embedding_current_state[X]

            elif norm_M < self.MinDist:
                self.embedding_next_state[X] = self.embedding_current_state[Y] - self.MinDist * M1

            else:
                self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - self.MinDist) * M1

class DependencyEmebeddingDocument_v23_07_2023_Power(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  MinDist :float = 1.0,
                  k : int = 2

                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.MinDist = MinDist
        self.k = k
        

    def iteration(self):
        
        for X in self.network.nodes:
            
            Y = -1
            while True:

                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break



            
            M = self.embedding_current_state[Y] - self.embedding_current_state[X]

            norm_M = np.linalg.norm(M)
            M1 = M / norm_M    

                        
            # if self.dependency_matrix[X][Y] == 0.0:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - 1.0 *  self.MinDist * M1 
            #     continue       

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]

            q = (D_x_y) * ((D_x_y + D_y_x) / 2)
                

            if norm_M == self.MinDist:
                self.embedding_next_state[X] = self.embedding_current_state[X]

            elif norm_M < self.MinDist:
                self.embedding_next_state[X] = self.embedding_current_state[Y] - self.MinDist * M1

            else:
                self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - self.MinDist) * M1

class GeneticEmbeddingAbsLogEachNeigh(GeneticEmbedding):
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

        for X in self.network.nodes:
            
            Y = -1
            while True:

                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break



            dependency_on_neigh = self.dependency_matrix[X][Y]

            dependency_reverse = self.dependency_matrix[Y][X]

            abs_diff = abs(dependency_on_neigh - dependency_reverse)

            step = self.max_step_portion * dependency_on_neigh * ((abs_diff+1)/2)
            
            neigh_emb = self.embedding_current_state[Y]
            current_node_emb = self.embedding_current_state[X]
            dir_vector = (neigh_emb - current_node_emb)
            dist_of_nodes = math.dist(current_node_emb,neigh_emb)

            repulsion = np.arctan(10* dist_of_nodes - 0.01)

            self.embedding_next_state[X] += dir_vector * repulsion * step 

class DependencyEmebeddingDocument_v23_07_2023_ProbBasedOnDep(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  MinDist :float = 1.0,
                  k : int = 2

                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.MinDist = MinDist
        self.k = k
        

    def iteration(self):
        
        for X in self.network.nodes:
            
            cumulative_probs = self.precomputed_dep_probs[X]

            random_number = random.random()
            Y = None

            for i, cumulative_prob in enumerate(cumulative_probs):
                if random_number <= cumulative_prob:
                    Y = i
                    if Y == X:
                        print("Vybrany rovnaky uzol")
                    break
            # Y = -1
            # while True:

            #     Y = random.randint(0,len(self.network.nodes)-1)
            #     if Y != X:
            #         break

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]

            norm_M = np.linalg.norm(M)

            if norm_M == 0.0: continue

            M1 = M / norm_M    


                        
            # if self.dependency_matrix[X][Y] == 0.0:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - 1.0 *  self.MinDist * M1 
            #     continue       

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]

            q = (D_x_y) * ((D_x_y + D_y_x) / 2)
                
            dist_of_nodes = math.dist(self.embedding_current_state[X],self.embedding_current_state[Y])

            repulsion = np.arctan(10* dist_of_nodes - 0.01)
            
            if norm_M == self.MinDist:
                self.embedding_next_state[X] = self.embedding_current_state[X]

            elif norm_M < self.MinDist:
                self.embedding_next_state[X] = self.embedding_current_state[Y] - self.MinDist * M1 * repulsion

            else:
                self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - self.MinDist) * M1 * repulsion

class DependencyEmebeddingDocument_v27_07_2023(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  eps : float

                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.eps = eps
        

    def iteration(self):
        
        for X in self.network.nodes:
            
            # cumulative_probs = self.precomputed_dep_probs[X]

            # random_number = random.random()
            # Y = None

            # for i, cumulative_prob in enumerate(cumulative_probs):
            #     if random_number <= cumulative_prob:
            #         Y = i
            #         if Y == X:
            #             print("Vybrany rovnaky uzol")
            #         break
            # Y = -1

            while True:

                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]

            norm_M = np.linalg.norm(M)

            #if norm_M == 0.0: continue

            M1 = M / norm_M    


                        
            # if self.dependency_matrix[X][Y] == 0.0:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - 1.0 *  self.MinDist * M1 
            #     continue       

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]


            
            q = max(D_x_y * (D_x_y + D_y_x) / 2.0 , self.eps)

            #minDist = max((1 - D_x_y) * (1 - (D_x_y + D_y_x) / 2), self.eps)
            minDist = 0.01

            # print("norm_M: ", norm_M)
            # print("minDist: ", minDist)
            # print("q: ", q)
            # print()
            # if norm_M < minDist:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - q * (minDist - norm_M) * M1
            if norm_M > minDist:
                self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - minDist) * M1

class DependencyEmebeddingDocument_v01_08_2023(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  eps : float

                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.eps = eps
        

    def iteration(self):
        
        for X in self.network.nodes:
            
            # cumulative_probs = self.precomputed_dep_probs[X]

            # random_number = random.random()
            # Y = None

            # for i, cumulative_prob in enumerate(cumulative_probs):
            #     if random_number <= cumulative_prob:
            #         Y = i
            #         if Y == X:
            #             print("Vybrany rovnaky uzol")
            #         break
            # Y = -1

            while True:

                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]

            norm_M = np.linalg.norm(M)

            #if norm_M == 0.0: continue

            M1 = M / norm_M    


                        
            # if self.dependency_matrix[X][Y] == 0.0:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - 1.0 *  self.MinDist * M1 
            #     continue       

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]


            
            q = max(D_x_y * (D_x_y + D_y_x) / 2.0 , self.eps)

            #minDist = max((1 - D_x_y) * (1 - (D_x_y + D_y_x) / 2), self.eps)
            minDist = 0.01

            # print("norm_M: ", norm_M)
            # print("minDist: ", minDist)
            # print("q: ", q)
            # print()
            # if norm_M < minDist:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - q * (minDist - norm_M) * M1
            if norm_M > minDist:
                self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - minDist) * M1
            
            # elif norm_M < minDist:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - q * (minDist - norm_M) * M1


class DependencyEmebeddingDocument_v01_08_2023_V2(GeneticEmbedding):
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

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  eps : float

                ):    
        super().__init__(network, dependency_matrix, embedding_dim)

        self.eps = eps
        
    def choose_node(self,X, p:float = 0.5):

        Y = None

        all_neighs = list(nx.neighbors(self.network,X))
        selected_neigh = random.choice(all_neighs)

        if random.random() < p:
            return selected_neigh

        else:
            while True:
                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break
            return Y 
            # neighs_of_neigh = list(nx.neighbors(self.network,selected_neigh))
            # neighs_of_neigh.remove(X)

            # if len(neighs_of_neigh) == 0:
            #     Y = None
            #     while True:
            #         Y = random.randint(0,len(self.network.nodes)-1)
            #         if Y != X:
            #             break
            #     return Y 

            # selected_neigh_of_neigh = random.choice(neighs_of_neigh)

            # return selected_neigh_of_neigh
        

    def iteration(self):
        
        for X in self.network.nodes:
            


            Y = self.choose_node(X, p = 0.2)

            #if X == Y: continue

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]

            norm_M = np.linalg.norm(M)

            #if norm_M == 0.0: continue

            M1 = M / norm_M    


                        
            # if self.dependency_matrix[X][Y] == 0.0:
            #     self.embedding_next_state[X] = self.embedding_current_state[X] - 1.0 *  self.MinDist * M1 
            #     continue       

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]


            
            q = max(D_x_y * (D_x_y + D_y_x) / 2.0 , self.eps)

            #minDist = max((1 - D_x_y) * (1 - (D_x_y + D_y_x) / 2), self.eps)
            minDist = 0.01

            if norm_M > minDist:
                self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - minDist) * M1
            
            elif norm_M < (minDist * self.eps):
                self.embedding_next_state[X] = self.embedding_current_state[X] - q * (minDist * self.eps - norm_M) * M1

class DependencyEmebeddingDocument_DepDist(GeneticEmbedding):

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  p : float,
                  depThreshold : float):
        super().__init__(network, dependency_matrix, embedding_dim)

        self.p = p

        ####################################
        N = len(self.network.nodes)
        self.indepDist = 0.01
        self.depDist = 0.0005
        self.depThreshold = depThreshold
        ####################################

    def choose_node(self,X):

        Y = None

        all_neighs = list(nx.neighbors(self.network,X))
        selected_neigh = random.choice(all_neighs)

        if random.random() < self.p:
            return selected_neigh

        else:
            while True:
                Y = random.randint(0,len(self.network.nodes)-1)
                if Y != X:
                    break
            return Y 
        
    def iteration(self):
        
        for X in self.network.nodes:

            Y = self.choose_node(X)

            M = self.embedding_current_state[Y] - self.embedding_current_state[X]
            norm_M = np.linalg.norm(M)
            M1 = M / norm_M    

            D_x_y = self.dependency_matrix[X][Y]
            D_y_x = self.dependency_matrix[Y][X]

            D_x_y = min( max( D_x_y, 0.05), 0.95)
            D_y_x = min( max( D_y_x, 0.05), 0.95)

            q = D_x_y * (D_x_y + D_y_x) / 2.0 

            DIST = None
            if D_x_y + D_y_x < self.depThreshold:
                DIST = self.indepDist
            else:
                DIST = self.depDist

            if norm_M > DIST:
                self.embedding_next_state[X] = self.embedding_current_state[X] + q * (norm_M - DIST) * M1
            
            if norm_M < self.depDist:
                self.embedding_next_state[X] = self.embedding_current_state[X] - q * (self.depDist - norm_M) * M1


class DepFinal(GeneticEmbedding):

    def __init__( self, network : nx.Graph,
                  dependency_matrix : np.ndarray,
                  embedding_dim : int,
                  
                ):    
        super().__init__(network, dependency_matrix, embedding_dim)


        ####################################
        self.neighProb = 0.2

        N = len(self.network.nodes)
        # self.maxDist = 0.01
        # self.minDist = 0.0005
        self.maxDist = 1.0
        self.minDist = 0.05

        #self.minDist = 0.0
        ####################################

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
