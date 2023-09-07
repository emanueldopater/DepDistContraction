import networkx as nx
import numpy as np

def export_to_gdf(filename : str,G : nx.Graph, embs : np.ndarray,has_labels = False ):


    f = open(filename, 'w')

    if has_labels:
        f.write('nodedef> name VARCHAR,label VARCHAR,x DOUBLE,y DOUBLE')
    else:
        f.write('nodedef> name VARCHAR,x DOUBLE,y DOUBLE')

    f.write('\n')


    for idx,node in enumerate(sorted(G.nodes)):        
        if has_labels:
            f.write(str(int(node)) + "," + "\"None\"," + str(embs[idx][0]) + "," +str(embs[idx][1]))
        else:
            f.write(str(int(node)) + ","  + str(embs[idx][0]) + "," + str(embs[idx][1]))

        f.write('\n')

    f.write('edgedef> node1,node2,weight DOUBLE,directed BOOLEAN')
    f.write('\n')
    for edge in G.edges:
        src, tar = edge

        edge_weight = G[src][tar]['weight']
        f.write(str(int(src)) + "," + str(int(tar)) + "," + str(edge_weight) + ",false")
        f.write('\n')
    f.close()


