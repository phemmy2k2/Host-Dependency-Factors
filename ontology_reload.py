import pickle, math, os
import pandas as pd

def compute_GO_features_Fisher(gopath,genepath,ppipath):
    import scipy.stats as stats

    print('Program Starts with converting GO data to dict...')
    # reads in pickle file that contains dict mapping of KO terms with genes
    GOterms = pd.read_csv(gopath, sep=',')
    # print(GOterms.head())
    print(GOterms.shape)
    # split and trim unwanted delimiters
    GOterms = pd.Series(GOterms.iloc[:,1])
    # GOterms = GOterms.str.lstrip('0123456789,')
    GOterms = GOterms.str.rstrip('\t\s')
    GOterms = GOterms.str.split('\t', n = 2, expand = True)
    GOterms[1] = GOterms[2].map(lambda x:[x.replace('\t',',')])
    GOgenes = {item[0]:item[1] for _,item in GOterms.iterrows()}
    cols = list(GOterms[0])
    col_len = len(cols)
    # print(GOgenes)

    print('Total no of GO terms is %d' % col_len)
    # convert PPI data to dict object
    print('Converting PPI file to dict ...')
    pklpath = ppipath.replace('.csv','.pkl')
    if os.path.exists(pklpath):
        # load pickle file
        with open(pklpath, 'rb') as df_PPI:
            # Ggo = open('input/fly/functional/ontology/pickles/KO_genes.pickle', 'rb')
            PPIgenes = pickle.load(df_PPI)  # all genes for a given GOterm
    else:
        # generate pickle file
        df_PPI = pd.read_csv(ppipath, sep=',', index_col=None)
        # PPI_terms = pd.Series(PPI_terms.iloc[:,0])
        # PPI_terms = PPI_terms.str.split(',', n = 1, expand = True)
        # creates a dict object and initialize with all elements as key and value
        print('Initializing PPI dict ...')
        PPIgenes = {x[0]: [x[0]] for _, x in df_PPI.iterrows()}
        PPIgenes2 = {x[1]: [x[1]] for _, x in df_PPI.iterrows()}
        PPIgenes.update(PPIgenes2)
        print('Converting PPI file to dict ...')
        for _,item in df_PPI.iterrows():
            item = list(item)
            PPIgenes[item[0]].append(item[1])
            PPIgenes[item[1]].append(item[0])
        # write object to pickle file for reuse
        fileobj = open(pklpath, 'wb')
        pickle.dump(PPIgenes, fileobj, protocol=2)
        fileobj.close()

    # reads in file that contains all the Dm genes
    print('Reading list of genes... Ensure there is no header in the file.. ')
    with open(genepath, 'r') as file:
        genes = file.readlines()  # all dm genes
        genes = set(list(genes))

    # N = len(genes)  # Total number of genes in the organism
    N = 11446         # REMOVE this line(used temporarily for additional genes)
    print('The Total number of genes is %d' %N)
    print('Computing score for each GO term...')
    dataset = {}   # genes represent keys while all the GOterms values represent the values
    count = 0

    for goterm in cols:
        count +=1
        GO_genes = GOgenes[goterm]
        GO_genes = GO_genes[0].split(',')
        M = len(GO_genes)
        print('%d out of %d GO term completed...' %(count,col_len))
        inner = 0
        for gene in genes:
            inner +=1
            gene = gene.strip() # removes spaces around current gene
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
            else: dataset[gene]= []
            # get the no of intersected genes between GO term and PPI-genes of the current gene
            a = len(set(GO_genes).intersection(PPI_genes))
            # print('The number of genes in both PPI neighbours and GOterm is %d' %a)

            # apply Fisher test
            b = n - a   # no of PPI genes
            c = M - a   # no of GO genes
            # d = N - n - b wrong
            d = N - b - c + a
            if d < 0:
                d = 0
            oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
            val = round(-1 * math.log(pvalue, 10), 4)
            dataset[gene].append(val)
            # print(count, inner)
    # exit()
    # Write dictionary to csv file
    # file_obj = pd.DataFrame(dataset, index=cols)
    file_obj = pd.DataFrame.from_dict(dataset, orient='index')
    file_obj.columns = cols
    # file_obj = file_obj.transpose()
    print(file_obj)
    print('Writing data to file...')
    outpath = gopath.replace('.tsv', '.result_additional.csv') #additional added to name
    file_obj.to_csv(outpath)

def extract_unique_genes_from_PPIfile(path):
    f = pd.read_csv(path)
    var1 = list(set(f['protein1'].values))
    print(len(var1))
    var2 = list(set(f['protein2'].values))
    print(len(var2))
    var = list(set(var1 + var2))
    series = pd.Series(var)
    path = path.replace('.csv','genelist.txt')
    series.to_csv(path, index=False, header=False)
    print(len(var))
    print('Completed!')

def covert_PPI_ids(path,dictpath):
    # convert Ids of PPI
    dictfile = pd.read_csv(dictpath, sep='\t', header=None)    #.iloc[:5,0:]
    print(dictfile.head())
    ppifile = pd.read_csv(path)
    # convert the dictfile to dict obj
    print('Converting dictfile to a dict object...')
    dictf = {v[0]:v[1] for _,v in dictfile.iterrows()}
    print(len(dictf))
    # exit()
    # print(dictf)
    print('Converting PPI Ids..')
    # loop through ppifile and populate pro1 and pro2 list with converted id
    pro1, pro2 = [], []
    uncoverted = set()

    for _, v in ppifile.iterrows():
        try:
            pro1.append(dictf[v[0]])
        except KeyError:
            pro1.append(v[0])
            print('%s on Left not converted' % v[0])
            uncoverted.add(v[0])
        try:
            pro2.append(dictf[v[1]])
        except KeyError:
            pro2.append(v[1])
            print('%s on Right not converted' % v[1])
            uncoverted.add(v[1])
    data = pd.DataFrame.from_records([pro1,pro2])
    data = data.transpose()
    data.columns = ['protein1', 'protein2']
    print('Writing data to file ...')
    path = path.replace('.csv', '_translated.csv')
    data.to_csv(path, index=None)
    print(uncoverted)
    print('Completed Successfully!')
    # map pro1 and pro2 into a dataframe and write to file

def test():
    f1 = [i for i in range(10)]
    f2 = [i for i in range(10)]
    dat = {}
    dat['col1'] = f1
    dat['col2'] = f2
    cols = ['pro1','pro2','pro3','pro4','pro5','pro6','pro7','pro8','pro9','pro10']
    data = pd.DataFrame(dat, index=cols)
    data = data.transpose()
    # data.columns = ['pro1','pro2','pro3','pro4','pro5','pro6','pro7','pro8','pro9','pro10']
    print(data)


if __name__ == "__main__":
    basepath = 'C:/Users/Femi/Desktop/Jena_stuff/obj2/data/'
    org = 'elegans'
    gopath = basepath + org +'/GotermR.tsv'
    genepath = basepath + org +'/added_gene_list.txt' #temp
    ppipath = basepath + org +'/ppi.csv'
    # dictppipath = basepath + org + '/protID_conv.tsv'
    # construct_coexpression_graph()
    # baba_plot()
    # hamm_distance()
    compute_GO_features_Fisher(gopath,genepath,ppipath)
    # covert_PPI_ids(ppipath, dictppipath)

    # test()
    exit()

