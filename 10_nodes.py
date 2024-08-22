import BN
from CyberBN import cyberbn


## Graph 1:
final_node = 'user(2)'

target_inf = cyberbn.inference(final_node)['T']

nodes = ['user(0)','<0,1>','<1, 2>'] 

print(cyberbn.model( 
    [cyberbn.read_infs("saved/1_10000_user(0)_pareto.csv"), 
     cyberbn.read_infs("saved/1_10000_<ssh, 1, 2>_pareto.csv"), 
     cyberbn.read_infs("saved/1_10000_<Exec, 0, 1>_pareto.csv")], 
     target_inf, nodes, 'normal', "In-degree")) 

# # Top K: selected based off of highest in degree
# nodes = ['<Dos, 0, 1>','<Exec, 0, 1>','<ssh, 1, 2>']




# print(cyberbn.model( 
#     [cyberbn.sensitivity_analysis(f'{nodes[0]}', final_node, 'normal'), 
#      cyberbn.sensitivity_analysis(f'{nodes[1]}', final_node, 'normal'), 
#      cyberbn.sensitivity_analysis(f'{nodes[2]}', final_node, 'normal')], 
#      target_inf, nodes, 'normal', "In-degree")) 

# print(cyberbn.model( 
#     [cyberbn.sensitivity_analysis(f'{nodes[0]}', final_node, 'pareto'), 
#      cyberbn.sensitivity_analysis(f'{nodes[1]}', final_node, 'pareto'), 
#      cyberbn.sensitivity_analysis(f'{nodes[2]}', final_node, 'pareto')], 
#      target_inf, nodes, 'pareto', "In-degree")) 


# # # # #top k: selected based off of highest out degree




# print(cyberbn.model( 
#     [cyberbn.sensitivity_analysis(f'{nodes[0]}', final_node, 'normal'), 
#      cyberbn.sensitivity_analysis(f'{nodes[1]}', final_node, 'normal'), 
#      cyberbn.sensitivity_analysis(f'{nodes[2]}', final_node, 'normal')], 
#      target_inf, nodes, 'normal', "In-degree")) 

# print(cyberbn.model( 
#     [cyberbn.sensitivity_analysis(f'{nodes[0]}', final_node, 'pareto'), 
#      cyberbn.sensitivity_analysis(f'{nodes[1]}', final_node, 'pareto'), 
#      cyberbn.sensitivity_analysis(f'{nodes[2]}', final_node, 'pareto')], 
#      target_inf, nodes, 'pareto', "In-degree")) 