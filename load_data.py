import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import util as util


def read_graphfile(datadir, dataname, max_nodes=None,isTox=False):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    
    # 处理有毒分子
    tox_node_label_num_dict = {'Tox21_AhR': 53, 'Tox21_AR': 53, 'Tox21_ARE': 54,'Tox21_AR-LBD': 53,'Tox21_aromatase': 53,'Tox21_ATAD5': 54,'Tox21_ER': 53,'Tox21_ER-LBD': 53,'Tox21_HSE': 54,'Tox21_MMP': 54,'Tox21_p53': 54,'Tox21_PPAR-gamma': 53}

    tox_edge_label_num_dict = {'Tox21_AhR': 4, 'Tox21_AR': 4,'Tox21_ARE': 4,'Tox21_AR-LBD': 4,'Tox21_aromatase': 4,'Tox21_ATAD5': 4,'Tox21_ER': 4,'Tox21_ER-LBD': 4,'Tox21_HSE': 4,'Tox21_MMP': 4,'Tox21_p53': 4,'Tox21_PPAR-gamma': 4}
    if(isTox):
        data_name=dataname.split('_')
        file_name=data_name[0]+'_'+data_name[1]
    
    
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1
    
    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    num_graph_labels = len(label_vals)

    label_map_to_int = {val: i for i, val in enumerate(np.sort(label_vals))}  # 映射为从0开始的
    # 改成标准的0,1；因为有些graph_label是-1,1 （-1对应0，
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])


    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]
    
    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
        G.graph['label'] = graph_labels[i - 1]
        for u in util.node_iter(G):
            # util.node_dict(G)[u]['sp_label'] = sp_label[u]
            if len(node_labels) > 0:
                if(isTox):
                    node_label_one_hot = [0] * tox_node_label_num_dict[file_name]
                else:
                    node_label_one_hot = [0] * num_unique_node_labels  
                # node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                util.node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                util.node_dict(G)[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        for n in util.node_iter(G):
            mapping[n] = it
            it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))

    max_num_nodes = max([G.number_of_nodes() for G in graphs])

    return graphs, num_graph_labels, max_num_nodes

