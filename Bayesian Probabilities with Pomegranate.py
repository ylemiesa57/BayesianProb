###Code from scratch model, copying the functionality of pomegranate
import numpy as np
import itertools

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cpts = {}

    def add_node(self, name, values):
        self.nodes[name] = values

    def add_edge(self, parent, child):
        if parent not in self.edges:
            self.edges[parent] = []
        self.edges[parent].append(child)

    def set_cpt(self, node, cpt):
        self.cpts[node] = cpt

    def get_parents(self, node):
        parents = []
        for parent, children in self.edges.items():
            if node in children:
                parents.append(parent)
        return parents

    #set the kwargs to what you want each variable to be in terms of state
    def joint_probability(self, **kwargs):
        prob = 1.0
        for node, value in kwargs.items():
            parents = self.get_parents(node)

            #if a parent node itself of the child node
            if not parents:
                prob *= self.cpts[node][value]

            #if child node
            else:
                parent_values = tuple(kwargs[parent] for parent in parents)
                prob *= self.cpts[node][parent_values][value]
        return prob

    # should return the probability for each state of a random variable based on its parent nodes and the relationships
    def inference (self, target_node, **kwargs):
        '''
        perform inference on one reandom variable at a time based on other variables..

        think about doing a few variables at a time? inference + joint probability?
        '''
        non_specific_nodes = []
        for node in self.nodes:
            if node != target_node and node not in kwargs:
                non_specific_nodes.append(node)

        # # possible combinations of states
        # possible_states = self.nodes[target_node]

        all_combos = list(itertools.product(*[self.nodes[var] for var in non_specific_nodes]))

        # Initialize a dictionary to hold the probabilities for each state of the target_node. build it up as we go
        target_probabilities = {state: 0.0 for state in self.nodes[target_node]}

        # Iterate over each state of the target_node
        for target_state in self.nodes[target_node]:
            # Iterate over each combination of unspecified variable states and try to 
            for combination in all_combos:
                # Create a full assignment for all variables.

                # add the known arguments for the variables we want to customize or set
                full_assignment = kwargs.copy()
                # add the target for inference that we want to predict the probability it is in a certain state?
                full_assignment[target_node] = target_state
                # update with the combos of all the variables that we know won't work
                full_assignment.update(dict(zip(non_specific_nodes, combination)))
                
                # Calculate the joint probability for this assignment. perform joint probability calculation
                joint_prob = self.joint_probability(**full_assignment)
                
                # Add the joint probability to the corresponding state of the target_node. iterating through the for loop updates the variables with probabilities.
                target_probabilities[target_state] += joint_prob
        

     # Normalize the probabilities to sum to 1
        total_prob = sum(target_probabilities.values())
        for state in target_probabilities:
            target_probabilities[state] /= total_prob

        return target_probabilities

        

# Initialize the Bayesian Network
bn = BayesianNetwork()

# Add nodes (Guest, Prize, Monty) and their values
bn.add_node('Guest', ['A', 'B', 'C'])
bn.add_node('Prize', ['A', 'B', 'C'])
bn.add_node('Monty', ['A', 'B', 'C'])

# Add edges
bn.add_edge('Guest', 'Monty')
bn.add_edge('Prize', 'Monty')

# Define the CPTs
guest_cpt = {'A': 1./3, 'B': 1./3, 'C': 1./3}
prize_cpt = {'A': 1./3, 'B': 1./3, 'C': 1./3}
monty_cpt = {
    ('A', 'A'): {'A': 0.0, 'B': 0.5, 'C': 0.5},
    ('A', 'B'): {'A': 0.0, 'B': 0.0, 'C': 1.0},
    ('A', 'C'): {'A': 0.0, 'B': 1.0, 'C': 0.0},
    ('B', 'A'): {'A': 0.0, 'B': 0.0, 'C': 1.0},
    ('B', 'B'): {'A': 0.5, 'B': 0.0, 'C': 0.5},
    ('B', 'C'): {'A': 1.0, 'B': 0.0, 'C': 0.0},
    ('C', 'A'): {'A': 0.0, 'B': 1.0, 'C': 0.0},
    ('C', 'B'): {'A': 1.0, 'B': 0.0, 'C': 0.0},
    ('C', 'C'): {'A': 0.5, 'B': 0.5, 'C': 0.0}
}

# Set the CPTs in the network
bn.set_cpt('Guest', guest_cpt)
bn.set_cpt('Prize', prize_cpt)
bn.set_cpt('Monty', monty_cpt)

# Example of computing a joint probability
jp = bn.joint_probability(Guest='A', Prize='B', Monty='C')
print(f'Joint Probability P(Guest=A, Prize=B, Monty=C): {jp}')

# Example of calculating inference
inf = bn.inference(target_node='Prize')
print(inf)


#checked by thinking about monty hall problem
inf1 = bn.inference(target_node='Prize', Guest='A', Monty='C')
print(inf1)
