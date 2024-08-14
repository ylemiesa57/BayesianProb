import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

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
        return [parent for parent, children in self.edges.items() if node in children]

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

    def inference(self, target_node, **kwargs):
        non_specific_nodes = [node for node in self.nodes if node != target_node and node not in kwargs]
        all_combos = self._generate_combinations(non_specific_nodes)

        target_probabilities = {state: 0.0 for state in self.nodes[target_node]}

        for target_state in self.nodes[target_node]:
            for combination in all_combos:
                full_assignment = kwargs.copy()
                full_assignment[target_node] = target_state
                full_assignment.update(dict(zip(non_specific_nodes, combination)))
                
                joint_prob = self.joint_probability(**full_assignment)
                target_probabilities[target_state] += joint_prob

        total_prob = sum(target_probabilities.values())
        for state in target_probabilities:
            target_probabilities[state] /= total_prob

        return target_probabilities

    def _generate_combinations(self, nodes):
        if not nodes:
            return [()]
        return [(val,) + combo
                for val in self.nodes[nodes[0]]
                for combo in self._generate_combinations(nodes[1:])]

    def sensitivity_analysis(self, target_node, distribution):
        num_runs = 1000
        noise_std = 0.1

        original_cpts = self.cpts.copy()
        ancestors = self._get_ancestors(target_node)

        inference_results = []
        for _ in range(num_runs):
            for node in ancestors:
                if node in self.cpts:
                    for state in self.cpts[node]:
                        if distribution == 'normal':
                            noise = stats.truncnorm(-1, 1, loc=0, scale=noise_std).rvs()
                        else:  # Pareto
                            noise = stats.pareto(1, scale=noise_std).rvs() - 1
                        
                        self.cpts[node][state] = np.clip(self.cpts[node][state] + noise, 0, 1)
                    
                    # Normalize
                    total = sum(self.cpts[node].values())
                    for state in self.cpts[node]:
                        self.cpts[node][state] /= total

            inference_result = self.inference(target_node)
            inference_results.append(inference_result['T'])

            # Reset CPTs
            self.cpts = original_cpts.copy()

        return inference_results

    def _get_ancestors(self, node):
        ancestors = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            parents = self.get_parents(current)
            for parent in parents:
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors

    def model(self, top_arrs, infs, nodes, distribution):
        df = pd.DataFrame({node: np.abs(np.array(arr) - inf) / 100 
                           for node, arr, inf in zip(nodes, top_arrs, infs)})

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df)
        plt.title(f'Inference Error due to {distribution.capitalize()} Noise')
        plt.xlabel('Node')
        plt.ylabel('Inference Error')
        plt.show()

        return 'Completed'

def sample_truncated(low, high, size=1):
    return np.clip(np.random.normal(loc=(low+high)/2, scale=(high-low)/4, size=size), low, high)