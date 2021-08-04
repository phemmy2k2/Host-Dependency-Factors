import os, pickle, sys
import pandas as pd

def generate_PPi_netx():
    import networkx as nx
    from graphrole import RecursiveFeatureExtractor #, RoleExtractor

    print('Program starts successfully...')
    from ast import literal_eval
    file = pd.read_csv('output/string_interactions_extract.csv')
    on_var = 'geneId'

    try:
        graph_obj = file.replace('_extract.csv', '_graph.pkl')
        if not os.path.exists(graph_obj):
            print('Graph generation starts')
                # edge_list = [literal_eval(i.strip('\n')) for i in file.readlines()]  # for graph generation
            edge_list = [literal_eval((str(row['protein1']))) for _, row in file.iterrows()]  # for graph generation
            # section below generates a graph
            print('Converting edges to graph...')
            G = nx.Graph()
            G.add_edges_from(edge_list)
            # save graph as an object
            print('Saving graph as pickle object... ')
            with open(graph_obj, 'wb') as f:
                pickle.dump(G, f)
        else:
            with open(graph_obj, 'rb') as f:
                G = pickle.load(f)
            print('Graph loaded from file')

        # section below computes the graph attributes
        # Number of genes with similar expression (Node degree)

        print('Computing closeness centrality feature...')
        cc = nx.closeness_centrality(G)
        cc = pd.DataFrame(list(cc.items()), columns=[on_var, 'cc'])

        print('Computing betweeness centrality feature...')
        bc = nx.betweenness_centrality(G)
        bc = pd.DataFrame(list(bc.items()), columns=[on_var, 'bc'])
        all = cc.merge(bc, how='inner', on=on_var)

        print('Computing eigenvector centrality feature...')
        ev = nx.eigenvector_centrality(G)
        ev = pd.DataFrame(list(ev.items()), columns=[on_var, 'ev'])
        all = all.merge(ev, how='inner', on=on_var)

        print('Computing clustering coefficient feature...')
        cco = nx.clustering(G)
        cco = pd.DataFrame(list(cco.items()), columns=[on_var, 'cco'])
        all = all.merge(cco, how='inner', on=on_var)

        # print('Computing clique feature...') # problematic feature
        # cliq = nx.algorithms.number_of_cliques(G, nodes=None, cliques=None)
        # cliq = pd.DataFrame(list(cliq.items()), columns=[on_var, 'cliq'])
        # all = all.merge(cliq, how='inner', on=on_var)

        print('Computing load centrality feature...')
        load = nx.load_centrality(G)
        load = pd.DataFrame(list(load.items()), columns=[on_var, 'load'])
        all = all.merge(load, how='inner', on=on_var)

        print('Computing subgraph centrality feature...')
        subg = nx.subgraph_centrality(G)
        subg = pd.DataFrame(list(subg.items()), columns=[on_var, 'subg'])
        all = all.merge(subg, how='inner', on=on_var)

        print('Computing harmonic centrality feature...')
        harm = nx.harmonic_centrality(G)
        harm = pd.DataFrame(list(harm.items()), columns=[on_var, 'harm'])
        all = all.merge(harm, how='inner', on=on_var)

        print('Computing pagerank feature...')
        p_rank = nx.pagerank(G)
        p_rank = pd.DataFrame(list(p_rank.items()), columns=[on_var, 'p_rank'])
        all = all.merge(p_rank, how='inner', on=on_var)
        all.set_index('geneId', inplace=True)

        print('\nGenerating refex features...')
        feature_extractor = RecursiveFeatureExtractor(G)
        features = feature_extractor.extract_features()

        features.index.name = 'geneId'
        print('Refex features Completed Successfully')
        print('\nJoining result from Networkx and Refex...')

        res = all.join(features, how='inner')
        print(res.head())
        print(res.shape)

        res.to_csv('topology_features.csv')
        print('Topology features successfully generated!')

    except: # catch *all* exceptions
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )


if __name__ == '__main__':
    print('Program running...')
    generate_PPi_netx()
    # merge_netx_refex()