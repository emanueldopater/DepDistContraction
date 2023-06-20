
import pandas as pd
import numpy as np
import networkx as nx


#load edgelist
def load_net_from_edge_list(path, sep = ',', header = None, names = ['source', 'target', 'weight'], has_edge_weights = True):
    if has_edge_weights:
        df = pd.read_csv(path, sep = sep, header = header, names = names)
    else:
        df = pd.read_csv(path, sep = sep, header = header, names = names[:2])
        df['weight'] = 1.0


    # initial label -> new label (id)
    node_id_mapping = {}

    initial_node_id = 0

    G = nx.Graph()


    for edge in df.values:
        source = edge[0]
        source_id = None

        target = edge[1]
        target_id = None

        weight = edge[2]

        # source mapping
        if source not in node_id_mapping:
            node_id_mapping[source] = initial_node_id
            source_id = initial_node_id

            G.add_node(source_id, initial_label = source)

            initial_node_id += 1
        else:
            source_id = node_id_mapping[source]

        # target mapping
        if target not in node_id_mapping:
            node_id_mapping[target] = initial_node_id
            target_id = initial_node_id
            G.add_node(target_id, initial_label = target)

            initial_node_id += 1
        else:
            target_id = node_id_mapping[target]

        G.add_edge(source_id, target_id, weight = weight)
    
    return G