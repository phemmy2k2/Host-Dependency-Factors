""" Code selects non-redundant features based on corr of the features with class label
 Outputs feature rank file contains ranked list of features based on corr with class label
 The first column must be row header """
from collections import OrderedDict
import pandas as pd
# import sklearn.datasets as data

### set variables ###
path = ''
coef = 0.5

##### End ##########
 ########## ########## ########## ########## ##########
# Needed only when there is need to exclude features from file
# with open(useless_features, 'r') as excl:
#     exclude_cols = [x.strip('\n') for x in excl]
# ds = pd.read_csv(path, usecols=lambda x: x not in exclude_cols)
########## ########## ########## ########## ##########
ds = pd.read_csv(path)
data = ds.iloc[:, 1:]
all_features = list(data)  # Get list of data features

corr = data.corr()  # Compute correlation coefficient of all features
cor_target = abs(corr["label"])  # Extract corr coef of class label
relevant_features = cor_target[cor_target > 0.0001]  # Extract all features with corr coef > 0.001

n_cols = len(data.columns)
n_relcols = len(relevant_features)

# read in relevant features from file
rank_file = pd.DataFrame(relevant_features)
rank_file = rank_file[1:]
rank_file.sort_values(by=['label'], ascending=False, inplace=True)
# print(rank_file)

rank_dict = {}
count = 0
optimal_features = set()
# convert the csv file to a dictionay of features:rank
for index, item in rank_file.iterrows():
    count += 1
    rank_dict[index] = count

rank_ordered_dict = OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))

# select features without corr > abs(0.7) with other features
# select feature with highest rank if it correlates with multiple features
keys = [x[0] for x in rank_ordered_dict.items()]
for key in keys:
    key_others = keys.copy()
    key_others.remove(key)  # Ensure the remove is local
    best_feature = key
    for other in key_others:
        if abs(corr[key][other]) > coef:
            if rank_dict[key] > rank_dict[other]:
                # remove key from optimal_features if it already exists
                optimal_features.remove(key) if key in optimal_features else None
                if rank_dict[best_feature] > rank_dict[other]:
                    optimal_features.remove(best_feature) if best_feature in optimal_features else None
                    best_feature = other
            else:
                continue
        else:
            continue
    optimal_features.add(best_feature)

with open(basepath + 'experiments/optimal_features/' + fname + '.txt', 'w') as optimal:
    optimal.write('FlybaseId' + '\n' + 'label' + '\n')
    for item in optimal_features: optimal.write(str(item) + '\n')

print('Total No of features is %d' % len(all_features))
print('No of Optimal features is %d' % len(optimal_features))
print('Optimal features')
print(optimal_features)
print('redundant features')
print(set(all_features).difference(set(optimal_features)))

########### Correlation Plot ######################
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns, fontsize=8)
    ax.set_yticklabels(data.columns, fontsize=8)
    plt.show()
########### Correlation Plot Ends ######################
# import pip
# pip.main(['install', '--upgrade', 'numpy'])