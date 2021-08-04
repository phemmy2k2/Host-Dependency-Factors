import pandas as pd
import os, pickle

def convert_PPI_to_GGI():
    # program converts PPI to GGI and and generate GGI tuple file required by topology_features.py

    ### Read in input files (PPI file and conversion file) ####
    f = pd.read_csv('output/string_interactions_extract.csv')
    f_table = pd.read_csv('gProfiler_idconverter.csv')[['initial_alias','converted_alias']]

    # print(f.head())
    # print(f_table.head())
    # exit()
    dict_table = {}
    for _, item in f_table.iterrows():
        # dict_table[item[0]] = item[1]
        dict_table[str(item['initial_alias']).lower()] = item['converted_alias']
    print('Protein-Gene map dictionary created successfully!')

    # create GGI file and PPI tuple file for graph object generation
    col1, col2 = [],[]
    print('Program generating GGI data...')
    with open('output/GGI_tuple.txt', 'w') as outFile:
        for _, item in f.iterrows():
            # create ppi tuple file for graph object generation
            try:
                v1 = dict_table[str(item['node1_string_id']).lower()]
                v2 = dict_table[str(item['node2_string_id']).lower()]
                # print(v1,v2)
                col1.append(v1)
                col2.append(v2)
                items = (v1, v2)
                # print(items)
                outFile.write(str(items) + '\n')
            except Exception as e:
                # print(e)
                pass
            # print('Protein not found in the dictionary')
    # exit()
    # converted PPI -GGI dataframe
    res = pd.DataFrame([col1,col2])
    res = res.transpose()
    res.columns = ['protein1', 'protein2']
    print(res.head())
    res.to_csv('output/Dm_GGI.csv', index=False)
    print('PPI to GGI conversion Completed - 2 output files generated!')

def convert_PPiFile_dict():
    # converts GGI file to GGI dictionary
    df_PPI = pd.read_csv('output/string_interactions_extract.csv')
    # creates a dict object and initialize with all elements as key and value
    print('Initializing PPI dictionary ...')
    PPIgenes = {x[0]: [x[0]] for _, x in df_PPI.iterrows()}
    PPIgenes2 = {x[1]: [x[1]] for _, x in df_PPI.iterrows()}
    PPIgenes.update(PPIgenes2)
    print('Converting PPI file to dictionary ...')
    for _, item in df_PPI.iterrows():
        item = list(item)
        PPIgenes[item[0]].append(item[1])
        PPIgenes[item[1]].append(item[0])
    # write object to pickle file for reuse
    with open('output/GGI_dict.pkl', 'wb') as fileobj:
        pickle.dump(PPIgenes, fileobj)
    print('GGI dictionary successfully created!')

def convert_gProfilerFile_dictObject():
    import pickle

    file = str(input("Enter name of the geneset-GOterm (.gmt) file: "))
    with open(file, 'r') as infile:
        df = infile.readlines()
        GOgenes = {}
        # GOgenes = {row[0].split(',')[0]: list(filter(None, row[0].split(',')[1:])) for _, row in df.iterrows()}
        for row in df:
            # print(row)
            # exit()
            key = row.split('\t')[0]
            val = list(filter(None, row.split('\t')[2:]))
            GOgenes[key] = val
            # if len(val) < 5: # exclude terms with genes less than 5
            #     # print(len(val))
            #     continue
            # else:
            #     GOgenes[key] = val
        # print(key)
        # print(val)

    with open('output/goterm_gene_mapping.pkl','wb') as f:
        pickle.dump(GOgenes, f)
    print('Total no of GO terms is %d' % len(GOgenes))

def generate_GO_features_Fisher():
    import scipy.stats as stats
    import math
    # requires convert_gProfilerFile_dictObject()
    # requires convert_PPiFile_dict()
    # requires 3 input file
    # output one file

    gopath='output/goterm_gene_mapping.pkl'
    genepath='output/unique_prots.csv'
    ppipath='output/GGI_dict.pkl'
    print('Loading input files.. ')
    # load pickle file
    if os.path.exists(gopath):
        with open(gopath, 'rb') as go:
            GOgenes = pickle.load(go)  # all genes for a given GOterm
        # print(GOgenes)
    else:print('Generate GOterms pickle')

    if os.path.exists(ppipath):
        with open(ppipath, 'rb') as ppi:
            PPIgenes = pickle.load(ppi)  # all genes for a given GOterm
    else:print('Generate PPI pickle')

    # reads in file that contains all the Dm genes
    genes = pd.read_csv(genepath)['0'].tolist()  #, header=None
    # genes = list(set(genes['geneId'].values))

    N = len(genes)  # Total number of genes in the organism
    # N = 12792  # REMOVE this line(used temporarily for additional genes)
    print('The Total number of genes is %d' % N)
    dataset = {}  # genes represent keys while all the GOterms values represent the values
    cols = GOgenes.keys()
    col_len = len(cols)
    print('The Total number of GOterms is %d' % len(cols))
    count = 0

    for goterm in cols:
        count += 1
        GO_genes = GOgenes[goterm]
        # print(GO_genes)
        # exit()
        # GO_genes = GO_genes.split(',')
        M = len(GO_genes)
        print('%d out of %d GO term completed...' % (count, col_len))
        inner = 0
        for gene in genes: #0-1000
            # print(gene)
            inner += 1
            # gene = gene.strip()  # removes spaces around current gene
            # retrieve all associated PPI genes to the current gene
            if gene in PPIgenes:
                PPI_genes = PPIgenes[gene]
            else:
                PPI_genes = []
            n = len(PPI_genes)
            # print('The number of neighbour genes to %s in PPI network is %d' %(gene, n))
            # initialize each gene with empty list
            if gene in dataset:
                pass
            else:
                dataset[gene] = []
            # get the no of intersected genes between GO term and PPI-genes of the current gene
            a = len(set(GO_genes).intersection(PPI_genes))
            # print('The number of genes in both PPI neighbours and GOterm is %d' %a)
            # exit()
            # apply Fisher test
            b = n - a  # no of PPI genes
            c = M - a  # no of GO genes
            d = N - b - c + a
            try:
                if d < 0:
                    val = 0
                    # print(a, b, c, d, N)
                    # print('d is less then zero')
                else:
                    oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
                    val = round(-1 * math.log(pvalue, 10), 4)
            except ValueError as e:
                print('error type: ', type(e))
                val = 0

            dataset[gene].append(val)
            # print(count, inner)
    # Write dictionary to csv file
    # file_obj = pd.DataFrame(dataset)
    # file_obj = file_obj.transpose()
    file_obj = pd.DataFrame.from_dict(dataset, orient='index')
    print(file_obj.shape)
    file_obj.index.name = 'geneId'
    file_obj.columns = cols

    print(file_obj.head())
    # exit()
    print('Writing data to file...')
    file_obj.to_csv('GOterms_features.csv')
    print('Completed!')


if __name__ == '__main__':
    print('Program is running...')
    # convert_PPI_to_GGI()
    # convert_PPiFile_dict()
    convert_gProfilerFile_dictObject()
    generate_GO_features_Fisher()

# 1. Get protien interaction data from STRING DB
# 2. Extract clean PPI data and get unique proteins using get_unique_protein()
# 3. Use the unique proteins from (2) to get gene equivalent from gProfiler
# 4. Use the output from (3) to convert clean PPI file from (2) to GGI file using convert_PPI_to_GGI()
# 4b.convert_PPI_to_GGI() also generates GGI tuple version used to generate Topology features
# 5. Convert the GGI file to dictionary object using convert_PPiFile_dict()
# 6. Download the gmt file for the organism from gProfiler and Convert to dictionary object using convert_gProfilerFile_dictObject()
# 7. Generate GO features using generate_GO_features_Fisher()