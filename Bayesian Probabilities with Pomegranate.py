# #Bayesian Probabilities with Pomegranate

# from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node

# import math

# '''
# Under the Monte Hall Problem

# Monty never opens the door chosen by the guest.
# Monty never opens the door with the prize behind it.
# If Monty has a choice (when neither the guest's door nor the prize's door), he picks randomly between the remaining doors.


# '''

# #defining the probability distribution of the chances for guests and prizes
# guest = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})
# prize = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})


# #the cpt table provides probabilities on P(monte's decision | where prize is put && where the guest picks)
# monty = ConditionalProbabilityTable(
#         [['A', 'A', 'A', 0.0],
#          ['A', 'A', 'B', 0.5],
#          ['A', 'A', 'C', 0.5],
#          ['A', 'B', 'A', 0.0],
#          ['A', 'B', 'B', 0.0],
#          ['A', 'B', 'C', 1.0],
#          ['A', 'C', 'A', 0.0],
#          ['A', 'C', 'B', 1.0],
#          ['A', 'C', 'C', 0.0],
#          ['B', 'A', 'A', 0.0],
#          ['B', 'A', 'B', 0.0],
#          ['B', 'A', 'C', 1.0],
#          ['B', 'B', 'A', 0.5],
#          ['B', 'B', 'B', 0.0],
#          ['B', 'B', 'C', 0.5],
#          ['B', 'C', 'A', 1.0],
#          ['B', 'C', 'B', 0.0],
#          ['B', 'C', 'C', 0.0],
#          ['C', 'A', 'A', 0.0],
#          ['C', 'A', 'B', 1.0],
#          ['C', 'A', 'C', 0.0],
#          ['C', 'B', 'A', 1.0],
#          ['C', 'B', 'B', 0.0],
#          ['C', 'B', 'C', 0.0],
#          ['C', 'C', 'A', 0.5],
#          ['C', 'C', 'B', 0.5],
#          ['C', 'C', 'C', 0.0]], [guest, prize])


# # each node is part of the DAG for the bayesian network. monty node is a child of the parent nodes guest and prize
# # therefore P(m, p, g) = P(m | p and g) * p(p) * p(g)
# # we simply just multiply the probabilities for each node
# '''
# When combined into the joint probability 
# P(m,p,g), it represents the likelihood of all three events happening together.
# '''
# s1 = Node(guest, name="guest")
# s2 = Node(prize, name="prize")
# s3 = Node(monty, name="monty")

# #init the model
# model = BayesianNetwork("Monty Hall Problem")
# #link the nodes with states in the model
# model.add_states(s1, s2, s3)
# model.add_edge(s1, s3)
# model.add_edge(s2, s3)
# #bake the model!
# model.bake()



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

cyberbn = BayesianNetwork()

cyberbn.add_node('<Dos, 0, 1>', ['T', 'F'])
cyberbn.add_node('<Exec, 0, 1>', ['T', 'F'])
cyberbn.add_node('user(1)', ['T', 'F'])
cyberbn.add_node('<1, 2>', ['T', 'F'])
cyberbn.add_node('ssh(2)', ['T', 'F'])
cyberbn.add_node('<ssh, 1, 2>', ['T', 'F'])
cyberbn.add_node('user(2)', ['T', 'F'])

# probability of the S=T, N=T, or L=T is based on random probability or simply how hard ech thing is to access
# service (s), connection (n), privilege (L) --> need to be satisfied as preconditions for vulnerability to be reached
# postconditions occur only when that vulnerability or one of the same level is exploited

cyberbn.add_edge('<Dos, 0, 1>', 'user(1)')
cyberbn.add_edge('<Exec, 0, 1>', 'user(1)')

cyberbn.add_edge('<1, 2>', '<ssh, 1, 2>')
cyberbn.add_edge('user(1)', '<ssh, 1, 2>')
cyberbn.add_edge('ssh(2)', '<ssh, 1, 2>')

cyberbn.add_edge('<ssh, 1, 2>', 'user(2)')

#####

#THESE NEED TO BE CALCULATED INDEPNEDENTLY!!! BUT FOR NOW WE ARE JUST GOING WITH THIS!!!!!
vul_dos_0_1_cpt = {'T': 1./2, 'F': 1./2}

vul_exec_0_1_cpt = {'T': 1./2, 'F': 1./2}


#edges assigned order match order of which column is which
priv_user_1_cpt = { ('T', 'T'): {'T': 0.93, 'F': .07},
                   ('T', 'F'): {'T': .093, 'F': .07},
                   ('F', 'T'): {'T': .093, 'F': .07},
                   ('F', 'F'): {'T': 0, 'F': 1} }

conn_1_2_cpt = {'T': 1./2, 'F': 1./2}

serv_ssh_2_cpt = {'T': 1./2, 'F': 1./2}


## vul calculator = p(v | s = T, N = T, L = T) = CVSS(v)/10
vul_ssh_1_2_cpt = { ('T', 'T', 'T'): {'T': .08, 'F': .092},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

priv_user_2_cpt = {('T'): {'T': 1, 'F': 0},
                   ('F'): {'T': 0, 'F': 1} }

cyberbn.set_cpt('<Dos, 0, 1>', vul_dos_0_1_cpt)
cyberbn.set_cpt('<Exec, 0, 1>', vul_exec_0_1_cpt)
cyberbn.set_cpt('user(1)', priv_user_1_cpt)
cyberbn.set_cpt('<1, 2>', conn_1_2_cpt)
cyberbn.set_cpt('ssh(2)', serv_ssh_2_cpt)
cyberbn.set_cpt('<ssh, 1, 2>', vul_ssh_1_2_cpt)
cyberbn.set_cpt('user(2)', priv_user_2_cpt)

inf = cyberbn.inference(target_node='<ssh, 1, 2>')
print(inf)




'''
Traceback (most recent call last):
  File "/Users/yaphetlemiesa/Documents/Summer Urop/Bayesian Probabilities with Pomegranate.py", line 276, in <module>
    inf = cyberbn.inference(target_node='<ssh, 1, 2>')
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yaphetlemiesa/Documents/Summer Urop/Bayesian Probabilities with Pomegranate.py", line 153, in inference
    joint_prob = self.joint_probability(**full_assignment)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yaphetlemiesa/Documents/Summer Urop/Bayesian Probabilities with Pomegranate.py", line 116, in joint_probability
    prob *= self.cpts[node][parent_values][value]
            ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
KeyError: ('T',)
'''


