import numpy as np
from scipy import stats
import itertools
import copy
import plotly.graph_objects as go
import pandas as pd

def sample_truncated(mean=0, std=1, low=0, high=1, interest='normal'):
    # Calculate the lower and upper bounds for the distribution
    a = (low - mean) / std
    b = (high - mean) / std
    
    # Generate a random sample
    if interest == 'normal':
        return float(stats.truncnorm.rvs(a, b, loc=mean, scale=std))
    elif interest == 'pareto':
        shape = std
        #to be able to customize later
        return float(stats.truncpareto.rvs(1, 2) - 1)
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
        self.nodes[name] = np.array(values)

    def add_edge(self, parent, child):
        if parent not in self.edges:
            self.edges[parent] = []
        self.edges[parent].append(child)

    def set_cpt(self, node, cpt):
        self.cpts[node] = {k: np.array(v) if isinstance(v, list) else v for k, v in cpt.items()}

    def get_parents(self, node):
        return np.array([parent for parent, children in self.edges.items() if node in children])

    def get_ind_ancestors(self, node, visited=None):
        if visited is None:
            visited = set()

        if node in visited:
            return set()

        visited.add(node)
        parents = self.get_parents(node)

        if len(parents) == 0:
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

            if len(parents) == 0:
                prob *= self.cpts[node][value]
            else:
                parent_values = tuple(kwargs[parent] for parent in parents)
                
                if len(parents) == 1:
                    prob *= self.cpts[node][parent_values[0]][value]
                else:
                    prob *= self.cpts[node][parent_values][value]
        return prob

    def inference(self, target_node, **kwargs):
        non_specific_nodes = np.array([node for node in self.nodes if node != target_node and node not in kwargs])

        all_combos = np.array(list(itertools.product(*[self.nodes[var] for var in non_specific_nodes])))

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
    
    def sensitivity_analysis(self, target_node, normal_or_pareto):
        num_runs = 1000
        noise_std = 1

        original_bn_cpts = self.copy_cpt()
        ind_ancestors = self.get_ind_ancestors(target_node)

        inference_results = np.zeros(num_runs)
        for round in range(num_runs):
            for node in ind_ancestors:
                if node in self.cpts:
                    for state in self.cpts[node]:
                        gen_noise = sample_truncated(0, noise_std, 0, 1, normal_or_pareto)
                        if normal_or_pareto == "pareto":
                            gen_noise = sample_truncated(std=noise_std, low=1, high=2, interest=normal_or_pareto)

                        pos_or_neg = np.random.normal(0, noise_std)
                        pos_or_neg = 1 if pos_or_neg >= 0 else -1

                        self.cpts[node][state] = np.clip(self.cpts[node][state] + pos_or_neg * gen_noise, 0, 1)

                total_prob = sum(self.cpts[node].values())
                for state in self.cpts[node]:
                    self.cpts[node][state] /= total_prob
                
                inference_result = self.inference(target_node)
                inference_results[round] = inference_result['T']

                print(target_node, node, round)

                self.cpts = copy.deepcopy(original_bn_cpts)

        return inference_results
    
    def model(self, top_arrs: list[list], infs: list[int], nodes: list[str], normal_or_pareto) -> str:
        top_arrs = np.array(top_arrs)
        infs = np.array(infs)

        diffs = np.abs(infs[:, np.newaxis] - top_arrs) / 100

        max_len = max(len(diff) for diff in diffs)
        diffs_padded = [np.pad(diff, (0, max_len - len(diff)), 'constant', constant_values=np.nan) for diff in diffs]

        df = pd.DataFrame(dict(zip(nodes, diffs_padded)))

        fig = go.Figure()
        for node in nodes:
            fig.add_trace(go.Box(
                y=df[node].dropna(),
                name=node
            ))

        fig.update_layout(
            title=f'Inference Error due to {normal_or_pareto[0].upper() + normal_or_pareto[1:]} Noise',
            xaxis_title='Rank of Centrality',
            yaxis_title='Inference Error',
        )

        fig.show()

        return 'Completed'