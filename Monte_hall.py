from BN import BayesianNetwork

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
