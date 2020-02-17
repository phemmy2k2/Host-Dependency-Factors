import pandas as pd
import networkx as nx
import os.path, pickle
#####################################
# Code extracts compoundID from compound.dat file
#####################################
#
def get_compounds():
    inputFile = 'C:/Users/OluFemi/Downloads/Compressed/Databases/Drosophila/fly PGDB/4.0.1.1.1/data/reactions.dat'
    outFile = 'output/network/all_compounds.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    compounds = set()  # The list object stores compounds extracted from input file
    with open(outFile, 'w') as output:
        for line in lines:
            if ('LEFT -' in line) or ('RIGHT -' in line):
                cmpd = line.split(' - ')[1]
                compounds.add(cmpd.strip())

        for item in compounds:
            output.write(item + '\n')

    print('Completed Successfully!')

#####################################
# Code extracts reactionID from reaction.dat file
#####################################
# A datatype that can ensure consistent ordering of the items should be used
# for var reactions instead of a list
def get_reactions():
    inputFile = 'C:/Users/OluFemi/Downloads/Compressed/Databases/Drosophila/fly PGDB/4.0.1.1.1/data/reactions.dat'
    outFile = 'output/network/all_reactions.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    reactions = []  # The list object stores compounds extracted from input file
    count = 0
    with open(outFile, 'w') as output:
        for line in lines:
            if 'UNIQUE-ID -' in line:
                rxn = line.split('UNIQUE-ID -')[1]
                reactions.append(rxn.strip())

        for item in reactions:
            count += 1
            output.write('R' + str(count) + ' ' + item + '\n')

    print('Completed Successfully!')


#####################################
# Code extracts reactionID mapped with Enzyme no from reaction.dat file
#####################################
# This code is eventually not useful instead get_reactions_with_genes() was used
def get_reactions_with_EC():
    # inputFile = 'C:/Users/OluFemi/Downloads/Compressed/Databases/Drosophila/fly PGDB/4.0.1.1.1/data/reactions.dat'
    inputFile = 'input/test.txt'
    outFile = 'output/network/all_reactions_with_EC.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    reactions = {}  # The list object stores compounds extracted from input file
    count = 0
    switch = False
    with open(outFile, 'w') as output:
        rxn, enz = '', ''
        for line in lines:
            if 'UNIQUE-ID -' in line:
                switch = True
                rxn = line.split('UNIQUE-ID -')[1].strip()
                continue
            if 'EC-NUMBER -' in line and switch:
                enz = line.split('EC-NUMBER -')[1].strip()
                continue
            if '//' in line and rxn != '':
                reactions[rxn] = enz
                switch = False
                rxn, enz = '', ''
                continue

        df = pd.DataFrame(list(reactions.items()), columns=['rxn','EC_no'])
        df.to_csv(outFile, index=False)

    print('Completed Successfully!')


#####################################
# Code extracts GeneID mapped with Enzyme no from enzrxns.dat file
#####################################
# Ordered_dict datatype should be used for reactions object
def get_reactions_with_genes():
    inputFile = 'C:/Users/OluFemi/Downloads/Compressed/Databases/Drosophila/fly PGDB/4.0.1.1.1/data/enzrxns.dat'
    # inputFile = 'input/test_enzrxns.txt'
    outFile = 'output/network/all_reactions_with_genes.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    reactions = {}  # The list object stores compounds extracted from input file
    switch = False
    with open(outFile, 'w') as output:
        rxn, enz = '', ''
        for line in lines:
            if 'UNIQUE-ID -' in line:
                switch = True
                continue
            if 'ENZYME -' in line and switch:
                enz = line.split('ENZYME -')[1].strip()
                temp = enz.split('-')
                enz = temp[0] if temp[1] == 'Monomer' else temp
                continue
            if 'REACTION -' in line and switch:
                rxn = line.split('REACTION -')[1].strip()
                continue
            if '//' in line and rxn != '':
                reactions[rxn] = enz
                switch = False
                rxn, enz = '', ''
                continue

        df = pd.DataFrame(list(reactions.items()), columns=['rxn', 'gene'])
        df.to_csv(outFile, index=False)

    print('Completed Successfully!')


#####################################
# Code output a refined list of compounds by removing cofactors
#####################################
# Takes output from get_compounds() and exclude set of hub compounds
def get_true_compounds():
    inputFile = 'output/network/all_compounds.txt'
    hubFile = 'input/hub.txt'
    outFile = 'output/network/all_true_compounds.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    file2 = open(hubFile, 'r')
    hubs = file2.readlines()
    file2.close()

    compounds = []  # The list object stores compounds extracted from input file
    count = 0
    with open(outFile, 'w') as output:
        for line in lines:
            if not line in hubs:
                count += 1
                compounds.append('C' + str(count) + ' ' + line)

        for item in compounds:
            output.write(item)

    print('Completed Successfully!')

#####################################
# Code output reaction.dat file that have no cofactor
#####################################
# Takes reactions.dat file as input, remove the cofactors and return reactions_no_cofactor.dat
def generate_reactionFile_without_cofactors():
    inputFile = 'C:/Users/OluFemi/Downloads/Compressed/Databases/Drosophila/fly PGDB/4.0.1.1.1/data/reactions.dat'
    # inputFile = 'input/test.txt'
    hubFile = 'input/hub.txt'
    outFile = 'output/network/reactions_no_cofactor.dat'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    file2 = open(hubFile, 'r')
    hubs = file2.readlines()
    file2.close()
    var = []
    with open(outFile, 'w') as output:
        for line in lines:
            if ('LEFT -' in line) or ('RIGHT -' in line):
                item = line.split(' - ')[1]
                if item in hubs:
                    continue
                else:
                    var.append(line)
                    continue
            else:
                var.append(line)

        for item in var:
            output.write(item)

    print('Completed Successfully!')

#####################################
# Code convert reaction name to reactionID in the edgelist_2.txt file
#####################################
#
def rxnName_to_rxnID():
    inputFile = 'output/network/edgeList.txt'
    rxnFile = 'output/network/all_reactions.txt'
    outFile = 'output/network/edgeList_byID.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    file = open(rxnFile, 'r')
    rxns = file.readlines()
    file.close()

    reactions = []
    count = 0
    with open(outFile, 'w') as output:
        for line in lines:
            line = line.split()
            left = line[0]
            right = line[1]
            leftId, rightId = '', ''
            for rxn in rxns:
                if left in rxn:
                    leftId = rxn.split()[0]
                    continue
                elif right in rxn:
                    rightId = rxn.split()[0]
                    continue
            reactions.append(leftId + ' ' + rightId)
        for item in reactions:
            count += 1
            output.write('E' + str(count) + ' ' + item + '\n')

    print('Completed Successfully!')

#####################################
# Code output a set of edges
#####################################
# Code requires output from remove_cofactor_from_reaction()
def generate_reaction_edges():
    # Get the list of compounds
    inputFile = 'output/network/reactions_no_cofactor.dat'
    compoundFile = 'output/network/all_true_compounds.txt'
    outFile = 'output/network/edgelist_2.txt'
    outFile2 = 'output/network/edgelist_code.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    file2 = open(compoundFile, 'r')
    compounds = file2.readlines()
    file2.close()

    # compounds = ['TRANS-RXN54L-3424']
    edgeList = []
    with open(outFile, 'w') as output:
        # For each compound, Loop thru the reaction file
        count = 0
        for compound in compounds:
            L = []  # Stores inflow reactions (reactions having the target compound as substrate)
            R = []  # Stores outflow reactions (reactions having the target compound as product)
            compound = ' ' + compound.split()[1] + '\n'
            for line in lines:
                if '//' in line:
                    rxnID = ''
                    rxnDir = 'LEFT-TO-RIGHT'
                    continue
                elif 'UNIQUE-ID -' in line:
                    rxnID = line.split('UNIQUE-ID -')[1]
                    continue
                elif 'REACTION-DIRECTION -' in line:
                    rxnDir = line.split('REACTION-DIRECTION -')[1]
                    rxnDir = rxnDir.strip()
                    continue
                # store the reactionID into either R or L depending on loc and the reaction-direction
                elif compound in line:
                    loc = line.split(' - ')[0] # stores LEFT (substrate) or RIGHT (product)
                    if (loc == 'LEFT' and rxnDir == 'LEFT-TO-RIGHT') or (loc == 'LEFT' and rxnDir == 'REVERSIBLE') or (loc == 'RIGHT' and rxnDir == 'RIGHT-TO-LEFT'):
                        L.append(rxnID.strip())
                        continue
                    elif (loc == 'RIGHT' and rxnDir == 'LEFT-TO-RIGHT') or (loc == 'RIGHT' and rxnDir == 'REVERSIBLE') or (loc == 'LEFT' and rxnDir == 'RIGHT-TO-LEFT'):
                        R.append(rxnID.strip())
                        continue
            # Map each item in L to every item in R
            for right in R:
                for left in L:
                    if left != right:
                        edge = right + ' ' + left
                        edgeList.append(edge)
        # store the mappings
        with open(outFile2, 'w') as output2:
            edgeList = set(edgeList)
            for item in edgeList:
                count += 1
                output.write('E' + str(count) + ' ' + item + '\n')
                vars = item.split(' ')
                var1 = convert_rxnCompName_to_rnxCompCode(vars[0])
                var2 = convert_rxnCompName_to_rnxCompCode(vars[1])
                output2.write('E' + str(count) + ' ' + var1 + ' ' + var2 + '\n')
    print('Completed Successfully!')

#####################################
# Code output a set of bipartite edges
#####################################
# Code reads the reaction.dat file to construct bipartite edges
# A bipartite edge is a connection between a substrate and its reaction
# rxnCode is the manually generated code for each reaction

def convert_rxnCompName_to_rnxCompCode(item):
    # function that coverts compound to compoundCode
    # Inputfile contains both compound and reaction items
    inputFile = 'input/all_reactions_compounds.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    rxnCode = ''
    for line in lines:
        if item in line:
            rxnCode = line.split(' ')[0]
            break
    return rxnCode

def generate_bipartite_edges():
    # Get the list of compounds
    inputFile = 'C:/Users/OluFemi/Downloads/Compressed/Databases/Drosophila/fly PGDB/4.0.1.1.1/data/reactions.dat'
    # inputFile = 'input/test.txt'
    outFile2 = 'output/network/bipartite_edgelist_code.txt'
    outFile = 'output/network/bipartite_edgelist.txt'
    hubsFile = 'input/hub.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    file2 = open(hubsFile, 'r')
    cofactors = file2.readlines()
    file2.close()

    L = []  # Stores inflow reactions (reactions having the target compound as substrate)
    R = []  # Stores outflow reactions (reactions having the target compound as product)
    edgeList = []
    rxnID = ''
    rxnDir = ''
    with open(outFile, 'w') as output:
        # For each compound, Loop thru the reaction file
        count = 0
        print(len(lines))
        for line in lines:
            if 'UNIQUE-ID -' in line:
                rxnID = line.split('UNIQUE-ID -')[1].strip('\n').strip()
                continue
            elif 'REACTION-DIRECTION -' in line:
                rxnDir = line.split('REACTION-DIRECTION -')[1]
                rxnDir = rxnDir.strip()
                if rxnDir == 'RIGHT-TO-LEFT':
                    R = [item for item in L]
                    L.clear()
                continue
            # store the reactionID into either R or L depending on loc and the reaction-direction
            elif 'LEFT ' in line:
                compound = line.split(' - ')[1]  # stores LEFT (substrate) or RIGHT (product)
                if (compound not in cofactors):
                    L.append(compound.strip())
                    continue
            elif ('RIGHT ' in line and rxnDir == 'LEFT-TO-RIGHT') or ('RIGHT ' in line and rxnDir == 'REVERSIBLE'):
                compound = line.split(' - ')[1]  # stores LEFT (substrate) or RIGHT (product)
                if (compound not in cofactors):
                    R.append(compound.strip())
                    continue
            elif 'RIGHT ' in line and rxnDir == 'RIGHT-TO-LEFT':
                compound = line.split(' - ')[1]  # stores LEFT (substrate) or RIGHT (product)
                if (compound not in cofactors):
                    L.append(compound.strip())
                    continue
            elif '//' in line :
                if (len(L) > 0 or len(R) > 0):
                    # Map each compound in L and R to previous reaction
                    for right in R:
                        edge = rxnID + ' ' + right
                        edgeList.append(edge)
                    for left in L:
                        edge = left + ' ' + rxnID
                        edgeList.append(edge)

                    # Prepare variable objects for current reaction
                    L, R = [], []
                    # create 2 list object for compound codes

                    rxnID = ''
                    rxnDir = 'LEFT-TO-RIGHT'
                    continue

        # store the mappings
        with open(outFile2, 'w') as output2:
            edgeList = set(edgeList)
            for item in edgeList:
                count += 1
                output.write('E' + str(count) + ' ' + item + '\n')
                vars = item.split(' ')
                var1 = convert_rxnCompName_to_rnxCompCode(vars[0])
                var2 = convert_rxnCompName_to_rnxCompCode(vars[1])
                output2.write('E' + str(count) + ' ' + var1 + ' ' + var2 + '\n')

    print('Completed Successfully!')

def generate_bipartite_edges_noDirection():
    # Get the list of compounds
    inputFile = 'C:/Users/OluFemi/Downloads/Compressed/Databases/Drosophila/fly PGDB/4.0.1.1.1/data/reactions.dat'
    # inputFile = 'input/test.txt'
    outFile2 = 'output/network/bipartite_edgelist_code_noDirection.txt'
    outFile = 'output/network/bipartite_edgelist_noDirection.txt'
    hubsFile = 'input/hub.txt'

    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    file2 = open(hubsFile, 'r')
    cofactors = file2.readlines()
    file2.close()

    compds = []
    edgeList = []
    rxnID = ''
    with open(outFile, 'w') as output:
        # For each compound, Loop thru the reaction file
        count = 0
        print(len(lines))
        for line in lines:
            if 'UNIQUE-ID -' in line:
                rxnID = line.split('UNIQUE-ID -')[1].strip('\n').strip()
                continue
            # store the compounds into compds
            elif ('LEFT ' in line) or ('RIGHT ' in line):
                compound = line.split(' - ')[1]  # stores LEFT (substrate) or RIGHT (substrate)
                if (compound not in cofactors):
                    compds.append(compound.strip())
                    continue
            elif '//' in line :
                if len(compds) > 0:
                    # Map each compound in L and R to previous reaction
                    for compd in compds:
                        edge = rxnID + ' ' + compd
                        edgeList.append(edge)

                    # Prepare variable objects for current reaction
                    compds = []
                    rxnID = ''
                    continue

        # store the mappings
        with open(outFile2, 'w') as output2:
            edgeList = set(edgeList)
            for item in edgeList:
                count += 1
                output.write('E' + str(count) + ' ' + item + '\n')
                vars = item.split(' ')
                var1 = convert_rxnCompName_to_rnxCompCode(vars[0])
                var2 = convert_rxnCompName_to_rnxCompCode(vars[1])
                output2.write('E' + str(count) + ' ' + var1 + ' ' + var2 + '\n')

    print('Completed Successfully!')

#####################################
# Code bipartite graph using networkx package
#####################################

def construct_bipartite_graph():
    from ast import literal_eval
    import matplotlib.pyplot as plt

    # inputFile = "output/network/bipartite_edgelist_code_noDirection.txt"
    inputFile = "output/network/bipartite_edgelist_code_intuples.txt"
    file = open(inputFile, 'r')
    edge_list = [literal_eval(i.strip('\n')) for i in file.readlines()]
    file.close()

    inputFile = "output/network/all_reactions.txt"
    file = open(inputFile, 'r')
    rxn_list = [i.split()[0] for i in file.readlines()]
    file.close()

    inputFile = "output/network/all_true_compounds.txt"
    file = open(inputFile, 'r')
    comp_list = [i.split()[0] for i in file.readlines()]
    file.close()

    # edge_list = ['E 1 A\n','E 2 B\n','E 3 C','E 4 D']
    # edge_list = [(i.split()[1],i.split()[2]) for i in edge_list]
    #
    # print(edge_list)
    # with open('output/network/bipartite_edgelist_code_intuples.txt', 'w') as outFile:
    #     for item in edge_list:
    #         outFile.write(str(item) + '\n')
    # print('completed')

    D = nx.Graph()
    D.add_nodes_from(rxn_list, bipartite = 0)
    D.add_nodes_from(comp_list, bipartite=1)
    D.add_edges_from(edge_list)
    print(nx.is_bipartite(D))
    print(nx.info(D))

    # nx.draw(D)
    # plt.show()

#####################################
# Computes features of a reaction graph using networkx package
#####################################

def construct_reaction_graph():
    from ast import literal_eval
    import matplotlib.pyplot as plt
    inputFile = "output/network/expression_intuples.txt"
    # inputFile = "input/graveley_0.9.csv" # input to generate tuple format of the input. should be commented after tuple is generated

    file = open(inputFile, 'r')
    edge_list = [i.strip('\n') for i in file.readlines()]  # for tuple generation, should be commented after tuple is generated
    file.close()
    #####################  Reads in text file and format rows into tuples ##################
    # section below generates tuple format of the edgelist, it should be commented after tuple is generated
    # edge_list = [(i.split()[1], i.split()[2]) for i in edge_list]
    # with open('output/network/edgelist_code_intuples.txt', 'w') as outFile:
    #     for item in edge_list:
    #         outFile.write(str(item) + '\n')
    # print('completed')
    ##################################################

    file = open(inputFile, 'r')
    edge_list = [literal_eval(i.strip('\n')) for i in file.readlines()] # for graph generation
    file.close()

    # section below generates a graph
    # G = nx.Graph()
    # G.add_edges_from(edge_list)
    #
    # # section below computes the graph attributes
    # # closeness_centrality
    # cc = nx.closeness_centrality(G)
    # cc = pd.DataFrame(list(cc.items()), columns=['rxnId','cc'])
    # cc.to_csv('output/network/nx/cc.csv', index= False)
    #
    # # # betweenness centrality
    # bc = nx.betweenness_centrality(G)
    # bc = pd.DataFrame(list(bc.items()), columns=['rxnId', 'bc'])
    # bc.to_csv('output/network/nx/bc.csv', index=False)
    #
    # # # Eigenvector centrality
    # ev = nx.eigenvector_centrality(G)
    # ev = pd.DataFrame(list(ev.items()), columns=['rxnId', 'ev'])
    # ev.to_csv('output/network/nx/ev.csv', index=False)
    #
    # # # eccentricity
    # # ec = nx.eccentricity(G) # Failed cos network is not connected
    # # ec = pd.DataFrame(list(ec.items()), columns=['rxnId', 'ec'])
    # # ec.to_csv('output/network/nx/ec.csv', index=False)
    #
    # # # node degree
    # deg = nx.degree(G)
    # deg = pd.DataFrame(list(deg), columns=['rxnId', 'deg'])
    # deg.to_csv('output/network/nx/deg.csv', index=False)
    #
    # # clustering coefficient
    # cco = nx.clustering(G)
    # cco = pd.DataFrame(list(cco.items()), columns=['rxnId', 'cco'])
    # cco.to_csv('output/network/nx/cco.csv', index=False)
    #
    # #
    # # nx.draw(D)
    # # plt.show()
    # print('Completed Successfully!')


#####################################
# Computes gene expression features from expression edgelist
#####################################
def get_max_corr_coeff():
    from ast import literal_eval

    inputFile = "output/network/expression_intuples.txt"
    file = open(inputFile, 'r')
    edge_list = [literal_eval(i.strip('\n')) for i in file.readlines()]  # for graph generation
    file.close()

    # section below generates a graph
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)

    out = {}
    for n, nbrs in G.adj.items():
        mxWt = max([wt['weight'] for nbr, wt in nbrs.items()])
        out[n] = mxWt
    df = pd.DataFrame(list(out.items()), columns=['FlybaseId', 'MCC'])
    df.to_csv('output/network/nx/max_corr_coeff.csv')

    print('Completed Successfully')

def construct_coexpression_graph():
    from ast import literal_eval
    on_var = 'Beetle'
    # inputFile = "C:/Users/Femi/Desktop/Collaborators/Thomas/fly/Used/STRING_FBpp_and_FBgn.csv"  # set location of STRING PPI file
    inputFile = 'input/beetle/topology/STRING_PPIfm.txt'
    tuple_obj = "input/beetle/topology/STRING_PPI_intuples.txt" # set location of tuple obj if it exists
    #####################  Reads in csv file and format rows into tuples ##################
    # The converted tuples is stored as pickle obj for networkX to use
    if not os.path.exists(tuple_obj):
        # df = pd.read_csv(inputFile, usecols=['gene1_ID','gene2_ID'], sep='\t')
        df = pd.read_csv(inputFile, header=None)
        print(df.head())
        with open(tuple_obj, 'w') as outFile:
            for index, row in df.iterrows():
                # item = (row['gene_FBgn1'], row['gene_FBgn2'], row['PCC'])
                item = (row[0], row[1])
                outFile.write(str(item) + '\n')
            print('Tuple conversion completed')
    #################### #################### #################### ####################
    try:
        graph_obj = 'input/beetle/topology/beetle_PPI_graph.pkl'
        if not os.path.exists(graph_obj):
            print('Graph generation starts')
            file = open(tuple_obj, 'r')
            edge_list = [literal_eval(i.strip('\n')) for i in file.readlines()]  # for graph generation
            file.close()

            # section below generates a graph
            print('Convert edges to graph...')
            G = nx.Graph()
            G.add_edges_from(edge_list)

            # save graph as an object
            print('Save graph as pickle object... ')
            f = open(graph_obj, 'wb')
            pickle.dump(G, f)
            f.close()
        else:
            f = open(graph_obj, 'rb')
            G = pickle.load(f)
            f.close()
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

        print('Computing degree centrality feature...')
        deg = nx.degree(G)
        deg = pd.DataFrame(list(deg), columns=[on_var, 'deg'])
        all = all.merge(deg, how='inner', on=on_var)

        print('Computing clustering coefficient feature...')
        cco = nx.clustering(G)
        cco = pd.DataFrame(list(cco.items()), columns=[on_var, 'cco'])
        all = all.merge(cco, how='inner', on=on_var)

        print('Computing clique feature...')
        cliq = nx.algorithms.number_of_cliques(G, nodes=None, cliques=None)
        cliq = pd.DataFrame(list(cliq.items()), columns=[on_var, 'cliq'])
        all = all.merge(cliq, how='inner', on=on_var)

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

        all.to_csv('input/beetle/topology/STRING_PPI.csv', index=None)

        print('Completed Successfully')

        # print('Computing reaching centrality feature...')
        # reach = [nx.local_reaching_centrality(G,n) for n in G]
        # reach = pd.DataFrame(reach, columns=['FlybaseId', 'reach'])

        # networkx.exception.NetworkXError: Graph not connected.
        # print('Computing Information centrality feature...')
        # info = nx.information_centrality(G)
        # info = pd.DataFrame(list(info.items()), columns=['FlybaseId', 'info'])

        # networkx.exception.NetworkXError: Graph not connected.
        # print('Computing current_flow_betweenness_centrality feature...')
        # bc_flow = nx.current_flow_betweenness_centrality(G)
        # bc_flow = pd.DataFrame(list(bc_flow.items()), columns=['FlybaseId', 'bc_flow'])

        # in _accumulate_percolation        KeyError: 'FBgn0014931'
        # print('Computing percolation_centrality feature...')
        # perco = nx.percolation_centrality(G)
        # perco = pd.DataFrame(list(perco.items()), columns=['FlybaseId', 'perco'])


    except nx.NetworkXError as error:
        all.to_csv('input/beetle/topology/STRING_PPI.csv', index=None)
        print(error)
        # edge_list = [i.strip('\n') for i in file.readlines()]  # for tuple generation, should be commented after tuple is generated


def feature_extractor():
    # Automatically extract features from expression graph data
    from graphrole import RecursiveFeatureExtractor, RoleExtractor

    graph_obj = 'input/fly/topology/fly_PPI_graph.pkl'
    if not os.path.exists(graph_obj):
        print("Run 'construct_coexpression_graph()' to generate graph object" )
        exit()
    else:
        f = open(graph_obj, 'rb')
        G = pickle.load(f)
        f.close()
        print('Graph loaded from file')

    ####### Code used to extract nodes in the graph#####
    # with open('input/fly/topology/nodes.txt', 'w') as out:
    #     for node in G.nodes:
    #         out.write(node + '\n')
    ####################################

    feature_extractor = RecursiveFeatureExtractor(G)
    print('Feature extraction starts!')
    features = feature_extractor.extract_features()
    # save the extracted features as pickle file
    f = open('input/fly/topology/PPI_features.pkl', 'wb')
    pickle.dump(features, f)
    f.close()

    print('Role extraction starts!')
    role_extractor = RoleExtractor(n_roles=None)
    role = role_extractor.extract_role_factors(features)
    # save the extracted roles as pickle file
    f = open('input/fly/topology/PPI_roles.pkl', 'wb')
    pickle.dump(role, f)
    f.close()
    print(role)

    features.to_csv('input/fly/topology/graphrole_PPI_features.csv', index=None)
    print('Completed Successfully!')

def class_distribution():
    # run test codes
    import pandas as pd
    path = 'output/all_features_aggregate_NAvalues.csv'
    # path = 'C:/Users/OluFemi/PycharmProjects/Machine_Learning/output/resampled_data.csv'
    df = pd.read_csv(path, usecols=['label'])
    res = df.groupby('label').size()
    # print(df.isnull().sum()) # output no of null values
    # print(df.count())      # no of items in the dataframe
    print(res)             # class distribution

def test():
    # run test codes
    import pandas as pd
    df = pd.read_csv('output/all_features_aggregate.csv', usecols=['FlybaseId', 'label'])
    res = df.groupby('label').size()
    # print(df.isnull().sum()) # output no of null values
    # print(df.count())      # no of items in the dataframe
    print(res)             # class distribution

    # Sample code for maximum correlation coefficient
    # import networkx as nx
    # FG = nx.Graph()
    # FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
    # out = {}
    # for n, nbrs in FG.adj.items():
    #     mxWt = max([wt['weight'] for nbr, wt in nbrs.items()])
    #     out[n] = mxWt
    # # df = pd.DataFrame(out.items())
    # # df.to_csv('filename.csv')
    # print(out)
    #     # for nbr, eattr in nbrs.items():
    #     #     wt = eattr['weight']
    #     #     if wt < 0.5: print('(%d, %d, %.3f)' % (n, nbr, wt))

def reformat_PPI_file():
    import sys
    path = 'C:/Users/Femi/Desktop/Collaborators/Thomas/beetle/7070.protein.links.v11.0.txt/7070.protein.links.v11.0.txt'
    df = pd.read_csv(path, header= 0, usecols=['protein1','protein2'], sep='\s+')
    # print(df.head())
    # line = [str(item[0]).strip("7070.").strip('-PA') + '-' + str(item[1]).strip("7070.").strip('-PA') for _, item in df.iterrows()]
    with open('input/fly/topology/STRING_PPIfm.txt', 'w') as f:
        for _, item in df.iterrows():
            a = str(item[0]).strip("7070.").strip('-PA')
            b = str(item[1]).strip("7070.").strip('-PA')
            out = a + ',' + b
            f.write(out + '\n')

    print('Completed Successfully!')

if __name__ == "__main__":
    # get_reactions_with_genes()
    # get_reactions_with_EC()
    # construct_reaction_graph()
    construct_coexpression_graph()
    # feature_extractor()
    # get_max_corr_coeff()
    # construct_bipartite_graph()
    # generate_bipartite_edges_noDirection()
    # generate_bipartite_edges()
    # generate_reaction_edges()
    # class_distribution()
    # reformat_PPI_file()
    # test()


# Outer join pandas: Returns all rows from both tables, join records from the left which have matching keys in the right table.
# Left outer join pandas: Return all rows from the left table, and any rows with matching keys from the right table.