
from BN import BayesianNetwork, sample_truncated

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

a = sample_truncated()

serv_Dos_1_cpt = {'T': a, 'F': 1-a}

b = sample_truncated()

priv_user_0_cpt = {'T': b, 'F': 1-b}

c = sample_truncated()

conn_0_1_cpt = {'T': c, 'F': 1-c}

d = sample_truncated()

serv_Exec_1_cpt = {'T': d, 'F': 1-d}

e = sample_truncated()

conn_1_2_cpt = {'T': e, 'F': 1-e}

f = sample_truncated()

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

# print("Parents of <Dos, 0, 1>:", cyberbn.get_parents('<Dos, 0, 1>'))

# print("Attack graph tests")

# inf = cyberbn.inference(target_node='<ssh, 1, 2>')
# print("Inference on <ssh, 1, 2>. Return prob dist: ", inf)

# print(cyberbn.get_ind_ancestors('<ssh, 1, 2>'))

# print(cyberbn.sensitivity_analysis('<ssh, 1, 2>'))
# print(cyberbn.sensitivity_analysis('<Dos, 0, 1>'))
# print(cyberbn.sensitivity_analysis('<Exec, 0, 1>'))

# print(cyberbn.sensitivity_analysis('<ssh, 1, 2>'))


# # Top K: selected based off of highest in degree
# nodes = ['<Dos, 0, 1>','<Exec, 0, 1>','<ssh, 1, 2>']

# print(cyberbn.model(
#     [cyberbn.sensitivity_analysis(f'{nodes[0]}', 'normal'), 
#      cyberbn.sensitivity_analysis(f'{nodes[1]}', 'normal'), 
#      cyberbn.sensitivity_analysis(f'{nodes[2]}', 'normal')],
#      [cyberbn.inference(target_node=f'{nodes[0]}')['T'],
#      cyberbn.inference(target_node=f'{nodes[1]}')['T'],
#      cyberbn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'normal'))

# # print(cyberbn.model(
# #     [cyberbn.sensitivity_analysis(f'{nodes[0]}', 'pareto'), 
# #      cyberbn.sensitivity_analysis(f'{nodes[1]}', 'pareto'), 
# #      cyberbn.sensitivity_analysis(f'{nodes[2]}', 'pareto')],
# #      [cyberbn.inference(target_node=f'{nodes[0]}')['T'],
# #      cyberbn.inference(target_node=f'{nodes[1]}')['T'],
# #      cyberbn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'pareto'))


#top k: selected based off of highest out degree
nodes = ['user(0)','<0,1>','<1, 2>']

print(cyberbn.model(
    [cyberbn.sensitivity_analysis(f'{nodes[0]}', 'normal'), 
     cyberbn.sensitivity_analysis(f'{nodes[1]}', 'normal'), 
     cyberbn.sensitivity_analysis(f'{nodes[2]}', 'normal')],
     [cyberbn.inference(target_node=f'{nodes[0]}')['T'],
     cyberbn.inference(target_node=f'{nodes[1]}')['T'],
     cyberbn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'normal'))

# print(cyberbn.model(
#     [cyberbn.sensitivity_analysis(f'{nodes[0]}', 'pareto'), 
#      cyberbn.sensitivity_analysis(f'{nodes[1]}', 'pareto'), 
#      cyberbn.sensitivity_analysis(f'{nodes[2]}', 'pareto')],
#      [cyberbn.inference(target_node=f'{nodes[0]}')['T'],
#      cyberbn.inference(target_node=f'{nodes[1]}')['T'],
#      cyberbn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'pareto'))


# test beteweenus ranked, test what the max diff for a noise infeernce dist is to see if it makes that much of a difference

print("###########")

