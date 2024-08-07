###Code from scratch model, copying the functionality of pomegranate
import numpy as np
from scipy import stats
import itertools
import copy

#normal dist: N(0,1)
#Heavy tailed dist: Pareto

def sample_truncated_normal(mean=0, std=1, low=0, high=1):
    # Calculate the lower and upper bounds for the distribution
    a = (low - mean) / std
    b = (high - mean) / std
    
    # Generate a random sample
    return float(stats.truncnorm.rvs(a, b, loc=mean, scale=std))

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cpts = {}

    def copy_cpt(self):
        return copy.deepcopy

    def add_node(self, name, values):
        self.nodes[name] = values

    def add_edge(self, parent, child):
        if parent not in self.edges:
            self.edges[parent] = []
        self.edges[parent].append(child)
        print(f"Edge added: {parent} -> {child}")

    def set_cpt(self, node, cpt):
        self.cpts[node] = cpt

    def get_parents(self, node):
        parents = []
        for parent, children in self.edges.items():
            if node in children:
                parents.append(parent)
        return parents

    def get_ind_ancestors(self, node, visited=None):
        if visited is None:
            visited = set()

        if node in visited:
            return set()

        visited.add(node)
        parents = self.get_parents(node)

        if not parents:
            return {node}

        ancestors = set()
        for parent in parents:
            ancestors.update(self.get_ind_ancestors(parent, visited))

        return ancestors
    
    def modify_ind_cpt(self, node, change):
        self.cpts[node]['T'] = (self.cpts[node]['T'] + change) % 1
        self.cpts[node]['F'] = 1 - self.cpts[node]['T']
        
    def joint_probability(self, **kwargs):
        prob = 1.0
        for node, value in kwargs.items():
            parents = self.get_parents(node)
            print(f"Node: {node}, Parents: {parents}, Value: {value}")

            if not parents:
                prob *= self.cpts[node][value]
            else:
                parent_values = tuple(kwargs[parent] for parent in parents)
                print(f"Parent values: {parent_values}")
                print(f"CPT keys for {node}: {list(self.cpts[node].keys())}")
                
                if len(parents) == 1:
                    prob *= self.cpts[node][parent_values[0]][value]
                else:
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


print("Monty Hall Tests!")

# Example of computing a joint probability
jp = bn.joint_probability(Guest='A', Prize='B', Monty='C')
print(f'Joint Probability P(Guest=A, Prize=B, Monty=C): {jp}')

# Example of calculating inference
inf = bn.inference(target_node='Prize')
print(inf)


#checked by thinking about monty hall problem
inf1 = bn.inference(target_node='Prize', Guest='A', Monty='C')
print(inf1)

print("############################")


cyberbn = BayesianNetwork()

cyberbn.add_node('Dos(1)', ['T', 'F'])
cyberbn.add_node('user(0)', ['T', 'F'])
cyberbn.add_node('<0,1>', ['T', 'F'])
cyberbn.add_node('Exec(1)', ['T', 'F'])

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

cyberbn.add_edge('Dos(1)', '<Dos, 0, 1>')
cyberbn.add_edge('user(0)', '<Dos, 0, 1>')
cyberbn.add_edge('<0,1>', '<Dos, 0, 1>')

cyberbn.add_edge('user(0)', '<Exec, 0, 1>')
cyberbn.add_edge('<0,1>', '<Exec, 0, 1>')
cyberbn.add_edge('Exec(1)', '<Exec, 0, 1>')


cyberbn.add_edge('<Dos, 0, 1>', 'user(1)')
cyberbn.add_edge('<Exec, 0, 1>', 'user(1)')

cyberbn.add_edge('<1, 2>', '<ssh, 1, 2>')
cyberbn.add_edge('user(1)', '<ssh, 1, 2>')
cyberbn.add_edge('ssh(2)', '<ssh, 1, 2>')

cyberbn.add_edge('<ssh, 1, 2>', 'user(2)')

#THESE NEED TO BE CALCULATED INDEPNEDENTLY!!! BUT FOR NOW WE ARE JUST GOING WITH THIS!!!!!

#generating the probs for independent probs using normal distribution

a = sample_truncated_normal()

serv_Dos_1_cpt = {'T': a, 'F': 1-a}

b = sample_truncated_normal()

priv_user_0_cpt = {'T': b, 'F': 1-b}

c = sample_truncated_normal()

conn_0_1_cpt = {'T': c, 'F': 1-c}

d = sample_truncated_normal()

serv_Exec_1_cpt = {'T': d, 'F': 1-d}

e = sample_truncated_normal()

conn_1_2_cpt = {'T': e, 'F': 1-e}

f = sample_truncated_normal()

serv_ssh_2_cpt = {'T': f, 'F': 1-f}

# vul_dos_0_1_cpt = {'T': 1./2, 'F': 1./2}

#p(v_dos| pre conditions all true) = CVSS(v) / 10. 
vul_dos_0_1_cpt = { ('T', 'T', 'T'): {'T': .53, 'F': .47},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

# vul_exec_0_1_cpt = {'T': 1./2, 'F': 1./2}

vul_exec_0_1_cpt = { ('T', 'T', 'T'): {'T': .08, 'F': .092},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }


#edges assigned order match order of which column is which
priv_user_1_cpt = { ('T', 'T'): {'T': 0.93, 'F': .07},
                   ('T', 'F'): {'T': .093, 'F': .07},
                   ('F', 'T'): {'T': .093, 'F': .07},
                   ('F', 'F'): {'T': 0, 'F': 1} }


## vul calculator = p(v | s = T, N = T, L = T) = CVSS(v)/10
vul_ssh_1_2_cpt = { ('T', 'T', 'T'): {'T': .08, 'F': .092},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

# Needed to add handling here because of the fact that the function joint_probability can't understand single parent node
# needed to add handling for this.
priv_user_2_cpt = {'T': {'T': 1, 'F': 0},
                   'F': {'T': 0, 'F': 1} }

cyberbn.set_cpt('Dos(1)', serv_Dos_1_cpt)
cyberbn.set_cpt('user(0)', priv_user_0_cpt)
cyberbn.set_cpt('<0,1>', conn_0_1_cpt)
cyberbn.set_cpt('Exec(1)', serv_Exec_1_cpt)

cyberbn.set_cpt('<Dos, 0, 1>', vul_dos_0_1_cpt)
cyberbn.set_cpt('<Exec, 0, 1>', vul_exec_0_1_cpt)
cyberbn.set_cpt('user(1)', priv_user_1_cpt)
cyberbn.set_cpt('<1, 2>', conn_1_2_cpt)
cyberbn.set_cpt('ssh(2)', serv_ssh_2_cpt)
cyberbn.set_cpt('<ssh, 1, 2>', vul_ssh_1_2_cpt)
cyberbn.set_cpt('user(2)', priv_user_2_cpt)

print("Parents of <Dos, 0, 1>:", cyberbn.get_parents('<Dos, 0, 1>'))

print("Attack graph tests")

inf = cyberbn.inference(target_node='<ssh, 1, 2>')
print("Inference on <ssh, 1, 2>. Return prob dist: ", inf)

print(cyberbn.get_ind_ancestors('<ssh, 1, 2>'))

print("###########")

