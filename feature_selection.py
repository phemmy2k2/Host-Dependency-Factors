import pandas as pd

basepath = 'C:/Users/Femi/Desktop/Jena_stuff/benchmark/server/experiments/'
def remove_redundant_features(path, fname,coef=0.7):
    print('Program starts...')
    """ Code selects non-redundant features based on corr of the features with class label
     Outputs feature rank file contains ranked list of features based on corr with class label
     The first column must be row header """
    from collections import OrderedDict

    print('Loading input data...')
    ds = pd.read_csv(path)
    data = ds.iloc[:, 1:]
    all_features = list(data)  # Get list of data features

    print('Computing correlation coefficient...')
    corr = data.corr()  # Compute correlation coefficient of all features
    cor_target = abs(corr["label"])  # Extract corr coef of class label
    relevant_features = cor_target[cor_target > 0.01]  # Extract all features with corr coef > 0.001

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
    print('Ranking features...')
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
    print('Writing optimal features to file...')
    with open(basepath  + fname + '.txt', 'w') as optimal:
        optimal.write('FlybaseId' + '\n' + 'label' + '\n')
        for item in optimal_features: optimal.write(str(item) + '\n')

    print('Total No of features is %d' % len(all_features))
    print('No of Optimal features is %d' % len(optimal_features))
    print('Optimal features')
    print(optimal_features)
    print('redundant features')
    print(set(all_features).difference(set(optimal_features)))

if __name__ == "__main__":
    file_name = 'ogee_label'
    opt_fname = 'lasso_ogee'
    path = basepath + 'labeled_data/' + file_name + '.csv'
    optimal_features = basepath + 'optimal_features/' + opt_fname + '.txt'  # Applicable to train_test()

    remove_redundant_features(path, file_name)