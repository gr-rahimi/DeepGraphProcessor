import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import shutil
import pickle



graph_nodes_min_train = 5
graph_nodes_max_train = 35 # not contain this number
dataset_size_train = 10000
edge_count_scale_train = 3 # the bigger, less edges

graph_nodes_min_test = 30
graph_nodes_max_test = 40 # not contain this number
dataset_size_test = 2000
edge_count_scale_test = 3 # the bigger, less edges

margin = 0.05
H, W = 256, 256
dpi = 128

def generate_dataset():
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
        graph_nodes_min = graph_nodes_min_train if is_train else graph_nodes_min_test
        graph_nodes_max = graph_nodes_max_train if is_train else graph_nodes_max_test
        edge_count_scale = edge_count_scale_train if is_train else edge_count_scale_test
        node_count = np.random.randint(graph_nodes_min,graph_nodes_max)
        edges_count = np.random.randint(0, node_count * (node_count - 1)/(2 * edge_count_scale) + 1)
        g = nx.gnm_random_graph(node_count, edges_count)
        np.random.seed(i)
        pos = nx.random_layout(g)

        pos[0] = (0,0)
        pos[1] = (1,1)
        pos[2] = (0,1)

        fig = plt.figure(frameon=False, figsize = (W/dpi,H/dpi),dpi= dpi) # dpi is 100 by default
        ax = fig.add_axes([0, 0, 1 , 1])
        ax.axis('off')
        ax.set_xlim([0 - margin, 1 + margin])
        ax.set_ylim([0 - margin , 1 + margin])

        nx.draw_networkx(g, ax = ax, pos = pos, node_size= [25]* node_count, with_labels=False)

        file_path = './dataset/'+ ('train/' if is_train else 'test/') + str(node_count)+\
                    '/g_' + str(i) + "_N_" + str(node_count) + "_E_" + str(edges_count)

        fig.savefig(file_path, pad_inches = 0)

        pos = pos.values() # from dictionary to list
        pos =[( (margin + (1 - y)) / (1 + 2 * margin), (x + margin)/(1 + 2 * margin) ) for (x , y) in pos ]
        with open(file_path+".pickle" , 'wb') as fp:
            pickle.dump(pos, fp)

        plt.close()
        if i % 10 == 0:
            print i, "/", dataset_size_train + dataset_size_test

if __name__ == "__main__":
    generate_dataset()