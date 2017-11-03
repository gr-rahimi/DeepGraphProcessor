import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import shutil

graph_nodes_min_train = 5
graph_nodes_max_train = 40 # not contain this number
dataset_size_train = 10000
edge_count_scale_train = 3 # the bigger, less edges

graph_nodes_min_test = 35
graph_nodes_max_test = 70 # not contain this number
dataset_size_test = 2000
edge_count_scale_test = 3 # the bigger, less edges


min_edge_count_train = graph_nodes_min_train * (graph_nodes_min_train - 1) / 2
min_edge_count_test = graph_nodes_min_test * (graph_nodes_min_test - 1) / 2

try:
    shutil.rmtree('./dataset')
except:
    pass

os.mkdir('./dataset')
os.mkdir('./dataset/train')
os.mkdir('./dataset/test')
for i in range(max(graph_nodes_max_train, graph_nodes_max_test )):
    os.mkdir("./dataset/train/" + str(i))
    os.mkdir('./dataset/test/'+ str(i))

for i in range(dataset_size_train + dataset_size_test):
    is_train = i < dataset_size_train
    graph_nodes_min = graph_nodes_min_train if is_train else graph_nodes_min_test
    graph_nodes_max = graph_nodes_max_train if is_train else graph_nodes_max_test
    edge_count_scale = edge_count_scale_train if is_train else edge_count_scale_test
    node_count = np.random.randint(graph_nodes_min,graph_nodes_max)
    edges_count = np.random.randint(0, node_count * (node_count - 1)/(2 * edge_count_scale))
    g = nx.gnm_random_graph(node_count, edges_count)
    nx.draw_random(g,node_size= [25]* node_count)
    plt.savefig('./dataset/'+ ('train/' if is_train else 'test/') + str(node_count)+ '/g_' + str(i) + "_N_" + str(node_count) + "_E_" + str(edges_count))
    plt.clf()
    if i % 10 == 0:
        print i, "/", dataset_size_train + dataset_size_test
