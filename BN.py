###Code from scratch model, copying the functionality of pomegranate
import numpy as np
from scipy import stats
import itertools
import copy
import plotly.graph_objects as go
import pandas as pd

#normal dist: N(0,1)
#Heavy tailed dist: Pareto

import scipy.stats as stats

def sample_truncated(mean=0, std=1, low=0, high=1, interest='normal'):
    # Calculate the lower and upper bounds for the distribution
    a = (low - mean) / std
    b = (high - mean) / std
    
    # Generate a random sample
    if interest == 'normal':
        return float(stats.truncnorm.rvs(a, b, loc=mean, scale=std))
    elif interest == 'pareto':
        shape = std
        return float(stats.truncpareto.rvs(a, b, loc=0, scale=1, size=1, random_state=None))
    else:
        raise ValueError("Invalid 'interest' parameter. Must be 'normal' or 'pareto'.")

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cpts = {}

    def copy_cpt(self):
        return copy.deepcopy(self.cpts)

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

            if not parents:
                prob *= self.cpts[node][value]
            else:
                parent_values = tuple(kwargs[parent] for parent in parents)
                
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
    
    def sensitivity_analysis(self, target_node, normal_or_pareto):
        #variables to set here
        num_runs = 1000
        noise_std = 1

        original_bn_cpts = self.copy_cpt()
        ind_ancestors = self.get_ind_ancestors(target_node)

        inference_results = []
        for round in range(num_runs):
            for node in ind_ancestors:
                #cautious check to make sure the cpts of node can be even updated
                if node in self.cpts:
                    for state in self.cpts[node]:
                        #select distribution here
                        gen_noise = sample_truncated(0,noise_std,0,1, normal_or_pareto)
                        if normal_or_pareto == "pareto":
                            gen_noise = sample_truncated(1,noise_std,1,5, normal_or_pareto)

                        pos_or_neg = np.random.normal(0, noise_std)
                        if pos_or_neg < 0:
                            pos_or_neg = -1
                        pos_or_neg = 1


                        #below just makes it so that the output state is always been 0 and 1
                        self.cpts[node][state] = min(max(self.cpts[node][state] + pos_or_neg*gen_noise, 0), 1)
                        


                #divide by sum to normalize to 1
                total_prob = sum(self.cpts[node].values())
                for state in self.cpts[node]:
                    self.cpts[node][state] /= total_prob
                
                #perform inference
                inference_result = self.inference(target_node)
                inference_results.append(inference_result['T'])

                print(target_node, node, round)

                #go back to orgiinal version of cpts
                self.cpts = copy.deepcopy(original_bn_cpts)

        return inference_results
    
    def model(self, top_arrs: list[list], infs: list[int], nodes: list[str], normal_or_pareto) -> str:

        #each input is a sim result prob for 'T' state

        #for each sim result, get the truth value
        diffs = []
        for i, selected in enumerate(top_arrs):
            diff_p = []
            for y, sim in enumerate(selected):
                #Calculate the corresponding errors as diff, abs, percents
                diff_p.append(abs(infs[i] - sim) * 1./100)
            diffs.append(diff_p)
        
        # at this point we have three lists of data points for three of our sets. plot.
        # diffs = [ [], [], []]


        print(len(diffs[0]),len(diffs[1]), len(diffs[2]))


        max_len = max([len(ele) for ele in diffs])

        for diff in diffs:
            if len(diff) < max_len:
                diff.extend([""]*(max_len-len(diff)))
        

        df = pd.DataFrame(dict(zip(nodes, diffs)))

        fig = go.Figure()
        for i, node in enumerate(nodes):
            fig.add_trace(go.Box(
                y=df[node],
                name=node
            ))

        fig.update_layout(
            title=f'Inference Error due to f{normal_or_pareto[0].upper() + normal_or_pareto[1:]} Noise',
            xaxis_title='Rank of Centrality (In degree- Left to Right)',
            yaxis_title='Inference Error'
        )

        fig.show()

        return 'Completed'





