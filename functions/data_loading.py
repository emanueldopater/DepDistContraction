
import pandas as pd
import numpy as np
import networkx as nx


# Reindex the nodes in edge list to start from 0 to better index as arrays.
def scale_edge_list(df):

    source_targer_values = df.iloc[:,:2].values
    edge_weights = df.iloc[:,2].values.reshape(-1,1)

    unique_values = sorted(np.unique(source_targer_values))

    unique_index = 0

    for unique_value in unique_values:
        source_targer_values = np.where(source_targer_values == unique_value, unique_index, source_targer_values)
        unique_index += 1
    
    # concat the scaled values with the edge weights
    scaled_edge_list = np.concatenate((source_targer_values, edge_weights), axis=1)   

    return scaled_edge_list



#load edgelist
def load_net_from_edge_list(path, sep = ',', header = None, names = ['source', 'target', 'weight'], has_edge_weights = True):
    if has_edge_weights:
        df = pd.read_csv(path, sep = sep, header = header, names = names)
    else:
        df = pd.read_csv(path, sep = sep, header = header, names = names[:2])
        df['weight'] = 1.0

    scaled_edge_list = scale_edge_list(df)


    G = nx.Graph()


    for edge in scaled_edge_list:
        #print(edge)
        source = int(edge[0])
        target = int(edge[1])

        weight = edge[2]

        G.add_edge(source, target, weight = weight)
    
    return G