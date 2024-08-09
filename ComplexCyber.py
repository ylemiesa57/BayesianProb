from BN import BayesianNetwork, sample_truncated

complex_bn = BayesianNetwork()

complex_bn.add_node('ssh(1)', ['T', 'F'])
complex_bn.add_node('(0,1)', ['T', 'F'])
complex_bn.add_node('ftp(1)', ['T', 'F'])
complex_bn.add_node('user(0)', ['T', 'F'])

complex_bn.add_node('ssh(0,1)', ['T', 'F'])
complex_bn.add_node('ftp_rhosts(0,1)', ['T', 'F'])

complex_bn.add_node('trust(0,1)', ['T', 'F'])
complex_bn.add_node('rsh(0,1)', ['T', 'F'])

complex_bn.add_node('user(1)', ['T', 'F'])
complex_bn.add_node('(1,2)', ['T', 'F'])
complex_bn.add_node('ssh(2)', ['T', 'F'])
complex_bn.add_node('(0,2)', ['T', 'F'])

complex_bn.add_node('ssh(1,2)', ['T', 'F'])
complex_bn.add_node('ssh(0,2)', ['T', 'F'])

complex_bn.add_node('user(2)', ['T', 'F']) 
complex_bn.add_node('local_bof(2)', ['T', 'F']) 
complex_bn.add_node('(2,2)', ['T', 'F'])
complex_bn.add_node('(0,2)', ['T', 'F'])

complex_bn.add_node('ssh(1,2)', ['T', 'F'])
complex_bn.add_node('ssh(0,2)', ['T', 'F'])

complex_bn.add_node('user(2)', ['T', 'F'])
complex_bn.add_node('local_bof(2)', ['T', 'F'])
complex_bn.add_node('(2,2)', ['T', 'F'])

complex_bn.add_node('bof(2,2)', ['T', 'F'])

#end goal
complex_bn.add_node('root(2)', ['T', 'F'])

#edges
complex_bn.add_edge('ssh(1)', 'ssh(0,1)')
complex_bn.add_edge('(0,1)', 'ssh(0,1)')
complex_bn.add_edge('user(0)', 'ssh(0,1)')

complex_bn.add_edge('(0,1)', 'ftp_rhosts(0,1)')
complex_bn.add_edge('ftp(1)', 'ftp_rhosts(0,1)')
complex_bn.add_edge('user(0)', 'ftp_rhosts(0,1)')

complex_bn.add_edge('ftp_rhosts(0,1)', 'trust(0,1)')

complex_bn.add_edge('(0,1)', 'rsh(0,1)')
complex_bn.add_edge('trust(0,1)', 'rsh(0,1)')
complex_bn.add_edge('user(0)', 'rsh(0,1)')

complex_bn.add_edge('ssh(0,1)', 'user(1)')
complex_bn.add_edge('rsh(0,1)', 'user(1)')

complex_bn.add_edge('user(1)', 'ssh(1,2)')
complex_bn.add_edge('(1,2)', 'ssh(1,2)')
complex_bn.add_edge('ssh(2)', 'ssh(1,2)')

complex_bn.add_edge('ssh(2)', 'ssh(0,2)')
complex_bn.add_edge('(0,2)', 'ssh(0,2)')
complex_bn.add_edge('user(0)', 'ssh(0,2)')

complex_bn.add_edge('ssh(1,2)', 'user(2)')
complex_bn.add_edge('ssh(0,2)', 'user(2)')

complex_bn.add_edge('user(2)', 'bof(2,2)')
complex_bn.add_edge('local_bof(2)', 'bof(2,2)')
complex_bn.add_edge('(2,2)', 'bof(2,2)')

complex_bn.add_edge('bof(2,2)', 'root(2)')


#THESE NEED TO BE CALCULATED INDEPNEDENTLY!!! BUT FOR NOW WE ARE JUST GOING WITH THIS!!!!!
a = sample_truncated(low=0.8, high=1)

serv_ssh_1_cpt = {'T': a, 'F': 1-a}

b = sample_truncated(low=0.8, high=1)

conn_0_1_cpt = {'T': b, 'F': 1-b}

c = sample_truncated(low=0.8, high=1)

serv_ftp_1_cpt = {'T': c, 'F': 1-c}

d = sample_truncated(low=0.8, high=1)

priv_user_0_cpt = {'T': d, 'F': 1-d}


###

vul_ssh_0_1_cpt = { ('T', 'T', 'T'): {'T': .1, 'F': .9},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

vul_ftp_rhosts_0_1_cpt = { ('T', 'T', 'T'): {'T': .8, 'F': .2},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

vul_trust_0_1_cpt = {'T': {'T': 1, 'F': 0},
                   'F': {'T': 0, 'F': 1} }

vul_rsh_0_1_cpt = { ('T', 'T', 'T'): {'T': .9, 'F': .1},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

###

# e = sample_truncated(low=0.8, high=1)

#edges assigned order match order of which column is which
priv_user_1_cpt = { ('T', 'T'): {'T': 1, 'F': 0},
                   ('T', 'F'): {'T': 1, 'F': 0},
                   ('F', 'T'): {'T': 1, 'F': 0},
                   ('F', 'F'): {'T': 0, 'F': 1} }

f = sample_truncated(low=0.8, high=1)

conn_1_2_cpt = {'T': f, 'F': 1-f}

g = sample_truncated(low=0.8, high=1)

serv_ssh_2_cpt = {'T': g, 'F': 1-g}

h = sample_truncated(low=0.8, high=1)

conn_0_2_cpt = {'T': h, 'F': 1-h}

###

vul_ssh_1_2_cpt = { ('T', 'T', 'T'): {'T': .1, 'F': .9},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

vul_ssh_0_2_cpt = { ('T', 'T', 'T'): {'T': .1, 'F': .9},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

###

i = sample_truncated(low=0.8, high=1)

# priv_user_2_cpt = {'T': i, 'F': 1-i}

priv_user_2_cpt = { ('T', 'T'): {'T': i, 'F': 1-i},
                   ('T', 'F'): {'T': i, 'F': 1-i},
                   ('F', 'T'): {'T': i, 'F': 1-i},
                   ('F', 'F'): {'T': 0, 'F': 1} }

j = sample_truncated(low=0.8, high=1)

serv_local_bof_2_cpt = {'T': j, 'F': 1-j}

k = sample_truncated(low=0.8, high=1)

conn_2_2_cpt = {'T': k, 'F': 1-k}

vul_bof_2_2_cpt = { ('T', 'T', 'T'): {'T': .1, 'F': .9},
                   ('T', 'T', 'F'): {'T': 0, 'F': 1},
                   ('T', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'T'): {'T': 0, 'F': 1},
                   ('T', 'F', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'T'): {'T': 0, 'F': 1},
                   ('F', 'T', 'F'): {'T': 0, 'F': 1},
                   ('F', 'F', 'F'): {'T': 0, 'F': 1} }

priv_root_2_cpt = {'T': {'T': 1, 'F': 0},
                   'F': {'T': 0, 'F': 1} }

complex_bn.set_cpt('ssh(1)', serv_ssh_1_cpt) 
complex_bn.set_cpt('(0,1)', conn_0_1_cpt) 
complex_bn.set_cpt('ftp(1)', serv_ftp_1_cpt) 
complex_bn.set_cpt('user(0)', priv_user_0_cpt ) 
complex_bn.set_cpt('ssh(0,1)', vul_ssh_0_1_cpt)
complex_bn.set_cpt('ftp_rhosts(0,1)', vul_ftp_rhosts_0_1_cpt)
complex_bn.set_cpt('trust(0,1)', vul_trust_0_1_cpt)
complex_bn.set_cpt('rsh(0,1)', vul_rsh_0_1_cpt) 
complex_bn.set_cpt('user(1)', priv_user_1_cpt) 
complex_bn.set_cpt('(1,2)', conn_1_2_cpt) 
complex_bn.set_cpt('ssh(2)', serv_ssh_2_cpt) 
complex_bn.set_cpt('(0,2)', conn_0_2_cpt) 
complex_bn.set_cpt('ssh(0,2)', vul_ssh_0_2_cpt)
complex_bn.set_cpt('ssh(1,2)', vul_ssh_1_2_cpt)
complex_bn.set_cpt('user(2)', priv_user_2_cpt)
complex_bn.set_cpt('local_bof(2)', serv_local_bof_2_cpt)
complex_bn.set_cpt('(2,2)', conn_2_2_cpt)
complex_bn.set_cpt('bof(2,2)', vul_bof_2_2_cpt)
complex_bn.set_cpt('root(2)', priv_root_2_cpt)

# Top 4: selected based off of highest in degree
nodes = ['rsh(0,1)', 'ssh(1,2)', 'ssh(0,2)', 'bof(2,2)']

print(complex_bn.model(
    [complex_bn.sensitivity_analysis(f'{nodes[0]}', 'normal'), 
     complex_bn.sensitivity_analysis(f'{nodes[1]}', 'normal'), 
     complex_bn.sensitivity_analysis(f'{nodes[2]}', 'normal')],
     [complex_bn.inference(target_node=f'{nodes[0]}')['T'],
     complex_bn.inference(target_node=f'{nodes[1]}')['T'],
     complex_bn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'normal'))

print(complex_bn.model(
    [complex_bn.sensitivity_analysis(f'{nodes[0]}', 'pareto'), 
     complex_bn.sensitivity_analysis(f'{nodes[1]}', 'pareto'), 
     complex_bn.sensitivity_analysis(f'{nodes[2]}', 'pareto')],
     [complex_bn.inference(target_node=f'{nodes[0]}')['T'],
     complex_bn.inference(target_node=f'{nodes[1]}')['T'],
     complex_bn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'pareto'))


#top 4: selected based off of highest in degree
nodes = ['trust(0,1)', 'local_bof(2)', 'user(2)', '(0,1)']

print(complex_bn.model(
    [complex_bn.sensitivity_analysis(f'{nodes[0]}', 'normal'), 
     complex_bn.sensitivity_analysis(f'{nodes[1]}', 'normal'), 
     complex_bn.sensitivity_analysis(f'{nodes[2]}', 'normal')],
     [complex_bn.inference(target_node=f'{nodes[0]}')['T'],
     complex_bn.inference(target_node=f'{nodes[1]}')['T'],
     complex_bn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'normal'))

print(complex_bn.model(
    [complex_bn.sensitivity_analysis(f'{nodes[0]}', 'pareto'), 
     complex_bn.sensitivity_analysis(f'{nodes[1]}', 'pareto'), 
     complex_bn.sensitivity_analysis(f'{nodes[2]}', 'pareto')],
     [complex_bn.inference(target_node=f'{nodes[0]}')['T'],
     complex_bn.inference(target_node=f'{nodes[1]}')['T'],
     complex_bn.inference(target_node=f'{nodes[2]}')['T']], nodes, 'pareto'))