
import pandas as pd
import numpy as np
import networkx as nx


# 
def scale_edge_list(df: pd.DataFrame) -> np.ndarray:
    """
    Reindex the nodes in edge list to start from 0 to better index as arrays. 

    Args:
        df (pd.DataFrame): _description_

    Returns:
        np.array: Reindexed edge list.
    """

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
def load_net_from_edge_list(path:str, 
                            sep: str = ',', 
                            header = None, 
                            names:list[str] = ['source', 'target', 'weight'], 
                            has_edge_weights: bool = True):
    """
    Load edge list 

    Args:
        path (str): Path to edge list file.
        sep (str, optional): Separator in csv file. 
        header (_type_, optional): Default is None because our csv file don't have headers. 
        names (list[str], optional): Name of columns for indexing edge weights if there are not present. 
        For our examples, don't change it.
        has_edge_weights (bool, optional): Whether edge list has edge weight. If yes, they are on 3-th place. 
        ('source', 'target', 'weight')

    Returns:
        nx.Graph: Constructed nx.Graph. 
    """
    if has_edge_weights:
        df = pd.read_csv(path, sep = sep, header = header, names = names)
    else:
        df = pd.read_csv(path, sep = sep, header = header, names = names[:2])
        df['weight'] = 1.0

    scaled_edge_list = scale_edge_list(df)


    G = nx.Graph()


    for edge in scaled_edge_list:
        source = int(edge[0])
        target = int(edge[1])

        weight = edge[2]

        G.add_edge(source, target, weight = weight)
    
    return G