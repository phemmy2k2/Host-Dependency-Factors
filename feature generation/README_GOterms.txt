Note:
The goterm_converted.py script requires that the PPI data from STRING database to be preprocessed. The clean_string_file.py script takes in as input the string_interactions.tsv file, cleans it and return string_interactions_extract.csv and a list of unique proteins required by the goterm script in the output directory. Download the geneset-GOterm (.gmt) file of the organism from gProfiler. 

Steps:
# 1. Get protien interaction data from STRING DB

# 2. Run clean_string_file.py to preprocess the PPI data from STRING DB, which also return a list of unique proteins contained in the PPI data

# 3. Use the unique proteins from (2) to get FBgn ID equivalent from gProfiler

# 4. Download the geneset-GOterm (.gmt) file of the organism from gProfiler

Ensure that the input files are in the same directory as the script and the following names are the same.

Rename the input files accordingly
# string_interactions.tsv == PPI file from STRINGDB
# gProfiler_idconverter.csv == the FBgn conversion of all proteins obtained from STRINGDB
# gprofiler_full_dmelanogaster.ENSG.gmt == geneset-GOterm (.gmt) file

A sample dataset (sample_genes.csv) is provided to test the code

The following packages are required for the scripts:
# pandas (pip install pandas)
# scipy

The takes time to complete depending on the number of query genes.

From the command prompt, run python <name of the file>.py, the final output will be located in the smae directory as the script