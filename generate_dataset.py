import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

graph_nodes_min = 5
graph_nodes_max = 40 # not contain this number
dataset_size = 100
edge_count_scale = 3 # the bigger, less edges

min_edge_count = graph_nodes_min * (graph_nodes_min - 1) / 2

for i in range(dataset_size):
    node_count = np.random.randint(graph_nodes_min,graph_nodes_max)
    if not os.path.exists("./dataset/"+ str(node_count)):
        os.makedirs("./dataset/"+ str(node_count))
    edges_count = np.random.randint(0, node_count * (node_count - 1)/(2 * edge_count_scale))
    g = nx.gnm_random_graph(node_count, edges_count)
    nx.draw(g,node_size= [25]* node_count)
    plt.savefig('./dataset/'+ str(node_count)+ '/g_' + str(i) + "_N_" + str(node_count) + "_E_" + str(edges_count))
    plt.clf()
