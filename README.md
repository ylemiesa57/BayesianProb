# Project working with Dr. Ranjan Pal

Developing bayesian network algorithm for inference and simulating cyber networks with a host privilege based attack graph

## future: AI applications


### Here's how we access it:

self.cpts[node] gives us the entire CPT dictionary for 'user(2)'
parent_values[0] gives us the state of the parent node ('<ssh, 1, 2>')
value is the state of 'user(2)' we're querying

So, self.cpts[node][parent_values[0]][value] might look like:
self.cpts['user(2)']['T']['T'], which would give us 1.

------------------------------

self.cpts[node] gives us the entire CPT dictionary for '<ssh, 1, 2>'
parent_values is a tuple of the states of all parent nodes, in order
value is the state of '<ssh, 1, 2>' we're querying

So, self.cpts[node][parent_values][value] might look like:
self.cpts['<ssh, 1, 2>'][('T', 'T', 'T')]['T'], which would give us 0.08.

--------------------------------
Let's say we have a network A -> B -> C -> D. What if we just asked for the joint probability P(B=?,C=?,D=?)
* We would have to compute using marginalization and conditional probabilities. Ex. below
P(B,C,D) = ∑[A] P(A,B,C,D) = ∑[A] P(D|C) * P(C|B) * P(B|A) * P(A)

* This is implemented in the Joint Probability function where we generate all possible combinations for missing nodes
if there all the nodes in the "kwargs" passed in aren't all the kwargs in the bayesian network.



Output:

Joint Probability: A single probability value.
Inference: A probability distribution for the target variable.

Key differences:

Output:

Joint Probability: A single probability value. (When values are unspecificed, it means we sum over the the prbobabilities of every state they are in.
this is the marginalize, NOT INTENDED PURPOSE THOUGH!!!)
Inference: A probability distribution for the target variable.


Marginalization:

Joint Probability: Doesn't try to marginalize over unspecified variables. Joint probability is for chance of certain conditions being met. If there is other variables unspecificed, then we just add up the probabilities for all their possibilities BUT NOT INDENDED INPUT!!!)
Inference: Marginalizes over all unspecified variables always.


Flexibility:

Joint Probability: Requires specifying values for all variables of interest.
Inference: Can handle partial information (evidence) and unknown variables.


Use case:

Joint Probability: Used when you want the probability of a specific scenario.
Inference: Used when you want to predict the probability distribution of a variable given some evidence.