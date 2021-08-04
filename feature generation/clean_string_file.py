import pandas as pd

def get_unique_protein():
    # extract needed columns from file downloaded from STRING DB
    #### This code will be used to read in the input data if complete PPI data was downloaded from STRING ###
    try:
        file = str(input("Enter name of the PPI file: "))
        dat = pd.read_csv(file, sep='\s', engine='python')
        cols = dat.columns.tolist()
        dat = dat[cols[:2]]
        ##### sample_PPI.txt
        pref = str(input("Enter the prefix before the gene Ids. E.g. 7091.: "))

        def strip(x):
            _pre = str(x).replace(pref, '')
            return _pre.split('-')[0]

        ## The header will be protein1 and protein2 if complete PPI data was used
        # dat = pd.read_csv('string_interactions.tsv', sep='\t')[['node1_string_id','node2_string_id']]
        dat = dat.applymap(strip)  # clean the raw data from STRING

        dat.to_csv('output/string_interactions_extract.csv',
                   index=False)  # write cleaned data to file for downstream analysis
        print(dat.head())

        # get unique protein from a PPI file
        allprots = dat[cols[0]].tolist()
        allprots.extend(dat[cols[1]].tolist())
        print('Number of protein in raw file %d' % len(allprots))
        allprots = set(allprots)
        print('Number of Unique protein in raw file %d' % len(allprots))
        res = pd.Series(list(allprots))
        res.to_csv('output/unique_prots.csv', index=False)

        print('Completed - 2 output files generated!')
    except:
        print("Enter valid input!")

def test():
    file = pd.read_csv('output/string_interactions_extract.csv')[:10]
    edge_list = [(str(row['protein1']).upper(), str(row['protein2']).upper()) for _, row in file.iterrows()]

    print(edge_list)

if __name__ == '__main__':
    print('Program is running...')
    # get_unique_protein()
    test()