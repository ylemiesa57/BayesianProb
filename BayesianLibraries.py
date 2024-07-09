#Bayesian Probabilities with Pomegranate

from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node

import math

'''
Under the Monte Hall Problem

Monty never opens the door chosen by the guest.
Monty never opens the door with the prize behind it.
If Monty has a choice (when neither the guest's door nor the prize's door), he picks randomly between the remaining doors.


'''

#defining the probability distribution of the chances for guests and prizes
guest = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})
prize = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})


#the cpt table provides probabilities on P(monte's decision | where prize is put && where the guest picks)
monty = ConditionalProbabilityTable(
        [['A', 'A', 'A', 0.0],
         ['A', 'A', 'B', 0.5],
         ['A', 'A', 'C', 0.5],
         ['A', 'B', 'A', 0.0],
         ['A', 'B', 'B', 0.0],
         ['A', 'B', 'C', 1.0],
         ['A', 'C', 'A', 0.0],
         ['A', 'C', 'B', 1.0],
         ['A', 'C', 'C', 0.0],
         ['B', 'A', 'A', 0.0],
         ['B', 'A', 'B', 0.0],
         ['B', 'A', 'C', 1.0],
         ['B', 'B', 'A', 0.5],
         ['B', 'B', 'B', 0.0],
         ['B', 'B', 'C', 0.5],
         ['B', 'C', 'A', 1.0],
         ['B', 'C', 'B', 0.0],
         ['B', 'C', 'C', 0.0],
         ['C', 'A', 'A', 0.0],
         ['C', 'A', 'B', 1.0],
         ['C', 'A', 'C', 0.0],
         ['C', 'B', 'A', 1.0],
         ['C', 'B', 'B', 0.0],
         ['C', 'B', 'C', 0.0],
         ['C', 'C', 'A', 0.5],
         ['C', 'C', 'B', 0.5],
         ['C', 'C', 'C', 0.0]], [guest, prize])


# each node is part of the DAG for the bayesian network. monty node is a child of the parent nodes guest and prize
# therefore P(m, p, g) = P(m | p and g) * p(p) * p(g)
# we simply just multiply the probabilities for each node
'''
When combined into the joint probability 
P(m,p,g), it represents the likelihood of all three events happening together.
'''
s1 = Node(guest, name="guest")
s2 = Node(prize, name="prize")
s3 = Node(monty, name="monty")

#init the model
model = BayesianNetwork("Monty Hall Problem")
#link the nodes with states in the model
model.add_states(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
#bake the model!
model.bake()
