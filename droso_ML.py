# code import csv file containing features from of drosophila data
# Featureset is normalized and trained on shallow ML algorithms
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing, svm, tree

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
# from sklearn.naive_bayes import GaussianNB
# from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn import metrics, neighbors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score,recall_score,f1_score,\
    roc_curve, precision_recall_curve, average_precision_score, roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits

# print(ds.head())
def train_test(path, optimal_features='', file_name=''):

    from imblearn.over_sampling import SMOTE

    bal_acc, accuracy_scores, mean_auc, precision_lst, recall_lst = [], [], [], [], []
    f1_lst, kappa_lst, fpr_lst, tpr_lst = [], [], [], []

    # load optimal features from file

    if optimal_features is not '':
        with open(optimal_features, 'r') as opt:
            optimal_cols = [x.strip('\n') for x in opt]
        ds = pd.read_csv(path, usecols=optimal_cols)
    else:
        ds = pd.read_csv(path)

    y = ds.pop('label')
    print(y.value_counts())
    n_features = len(list(ds))
    print('Total no of features is %d' %n_features)
    ds = ds.iloc[:, 1:]

    # convert values to float
    numpyMatrix = ds.values.astype(float)
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    X = scaler.fit_transform(numpyMatrix)

    # X = preprocessing.scale(ds)


    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    # clf = SVC(kernel='rbf', gamma='auto', probability=True)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    print('Training of classifier starts...')
    for train_index, test_index in sss.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
        clf.fit(X_resampled, y_resampled)
        # clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        ### Performance Evaluation Metrics #############
        accuracy = accuracy_score(y_test, y_pred)
        bal_accuracy = balanced_accuracy_score(y_test, y_pred)
        preci_score = precision_score(y_test, y_pred)  # average='macro' or 'micro' or 'weighted'
        recal_score = recall_score(y_test, y_pred)  # average='macro'(default) or 'micro' or 'weighted'
        f_score = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)   # The Kappa statistic varies from 0 to 1
        # compute AUC metric for each loop CV fold
        probs = clf.predict_proba(x_test)[::, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
        roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'
        # roc_auc = roc_auc_score(y_test, probs)
        print("AUC (fold): %f" % (roc_auc))
        # print('Cohens kappa: %f' % kappa)
        #####################################################

        mean_auc.append(roc_auc)
        accuracy_scores.append(accuracy)
        bal_acc.append(bal_accuracy)
        precision_lst.append(preci_score)
        recall_lst.append(recal_score)
        f1_lst.append(f_score)
        kappa_lst.append(kappa)
        fpr_lst.append(fpr)
        tpr_lst.append(tpr)

    print('\n')
    print("Mean AUC: %f" % np.mean(mean_auc))
    print('Mean Accuracy', np.mean(accuracy_scores))
    print('Mean Balanced Accuracy', np.mean(bal_acc))
    print("Mean Precision: %f" % np.mean(precision_lst))
    print('Mean Recall', np.mean(recall_lst))
    print('Mean F_measure', np.mean(f1_lst))
    print('Mean Kappa', np.mean(kappa_lst))
    ###################################################
    print('Plotting charts...')
    ############### Plot ROC curve ###################
    plt.title('Receiver Operating Characteristic')
    # fpr = np.mean(fpr_lst)
    # tpr = np.mean(tpr_lst)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    ############### Plot traning and test data ###################
    # plt.scatter(y_test, y_pred)
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    #
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ###################################################
    ############### Plot precision-recall chart ###################
    # from inspect import signature
    # precision, recall, _ = precision_recall_curve(y_test, y_pred)
    # average_precision = average_precision_score(y_test, y_pred)
    # print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    #
    # # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    # step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    # plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    # plt.show()
    ###################################################
    ############### Plot traning and test data ###################
    # plot loss during training
    # plt.subplot(211)
    # plt.title('Loss')
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # # plot accuracy during training
    # plt.subplot(212)
    # plt.title('Accuracy')
    # plt.plot(history.history['acc'], label='train')
    # plt.plot(history.history['val_acc'], label='test')
    # plt.legend()
    # plt.show()

    ###################################################
    # print('Dumping classifier into a pickle file...')
    # path = 'output/fly/classifier/' + file_name + '.pkl'
    # file_obj = open(path, 'wb')
    # pickle.dump(clf, file_obj)
    # file_obj.close()
    ###################################################
    print('\nSuccessfully Completed!')

def train_test_beta(ds, test_data):
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    print('Initializing and Preprocessing data ...')

    # bal_acc, accuracy_scores, mean_auc, precision_lst, recall_lst = [], [], [], [], []
    # f1_lst, kappa_lst, fpr_lst, tpr_lst = [], [], [], []

    y = ds.pop('label')
    print('Class label distribution %s' % Counter(y))

    print('Total Number of features %d' %len(list(ds)))
    ds = ds.iloc[:, 1:]

    y_test = test_data.pop('label')
    X_test = test_data.iloc[:, 1:]

    X = preprocessing.scale(ds)
    X_train, y_train = SMOTE().fit_resample(X, y)
    # clf = SVC(kernel='rbf', gamma='auto', probability=True)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")

    print('Training of classifier starts...')
    y_pred_train = cross_val_score(clf,X_train, y_train, cv=10)
    y_pred_test = cross_val_predict(estimator=clf, X = X_test, y = y_test, cv=3)

    accuracy = accuracy_score(y_test, y_pred_test)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred_test)
    preci_score = precision_score(y_test, y_pred_test)  # average='macro' or 'micro' or 'weighted'
    recal_score = recall_score(y_test, y_pred_test)  # average='macro'(default) or 'micro' or 'weighted'
    f_score = f1_score(y_test, y_pred_test)
    kappa = cohen_kappa_score(y_test, y_pred_test)  # The Kappa statistic varies from 0 to 1
    # compute AUC metric for each loop CV fold
    # probs = clf.predict_proba(X_test)[::, 1]
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
    # roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'

    # print("AUC (fold): %f" % (roc_auc))
    print('Accuracy %f' % accuracy)
    print('Balanced Accuracy %f' % bal_accuracy)
    print("Precision: %f" % preci_score)
    print('Recall %f' % recal_score)
    print('F_measure %f' % f_score)
    print('Kappa %f' % kappa)

    print('Accuracy score from train data %f' %np.mean(y_pred_train))
    # print(np.mean(y_pred_test))
    # print('\nSuccessfully Completed!')

def train_test_simple(ds, test_data):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from collections import Counter

    y_test = test_data.pop('label')
    X_test = test_data.iloc[:, 1:]

    y = ds.pop('label')
    print(y.value_counts())
    df = ds.iloc[:, 1:]
    print('Rescaling Input data...')
    X = preprocessing.scale(df)
    X_test = preprocessing.scale(X_test)

    # names = df.columns
    # Create the Scaler object
    # scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    # scaled_df = scaler.fit_transform(df)
    # scaled_df = pd.DataFrame(scaled_df, columns=names)

    X_train, y_train = SMOTE().fit_resample(X, y)
    # rus = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = rus.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_train))
    print('Fitting model to data...')
    # X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    # Perform 6-fold cross validation
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10)
    print('Accuracy score %f' %np.mean(scores))
    predictions = cross_val_predict(clf, X_train, y_train, cv=5)
    print('Prediction score %f' %np.mean(predictions))

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    ### Performance Evaluation Metrics #############
    accuracy = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    preci_score = precision_score(y_test, y_pred)  # average='macro' or 'micro' or 'weighted'
    recal_score = recall_score(y_test, y_pred)  # average='macro'(default) or 'micro' or 'weighted'
    f_score = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)  # The Kappa statistic varies from 0 to 1
    # compute AUC metric for each loop CV fold
    probs = clf.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
    roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'
    # roc_auc = roc_auc_score(y_test, probs)
    print("Mean AUC: %f" % roc_auc)
    print('Mean Accuracy %f' %accuracy)
    print('Mean Balanced Accuracy %f' %bal_accuracy)
    print("Mean Precision: %f" % preci_score)
    print('Mean Recall %f' %recal_score)
    print('Mean F_measure %f' %f_score)
    print('Mean Kappa %f' %kappa)

def train_test_basic(ds):
    print('Initializing and Preprocessing data ...')
    bal_acc, accuracy_scores, mean_auc, precision_lst, recall_lst = [], [], [], [], []
    f1_lst, kappa_lst, fpr_lst, tpr_lst = [], [], [], []
    # tpr_lst, fpr_lst = [], []
    y = ds.pop('label')
    print(y.value_counts())
    ds = ds.iloc[:, 1:]
    # print(len(list(ds)))
    X = preprocessing.scale(ds)
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)

    # n_neighbors = 6
    # X = (X / 16).astype(np.float32)  # Deep belief classifier requires strict float for all values

    print('Fitting model to the data in 10 CV folds ...')
    # clf = SVC(kernel='rbf', gamma='auto', probability=True)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    # clf = LogisticRegression().fit(x_train, y_train)
    # clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, solver='saga',
    #                          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    # clf = XGBClassifier()
    # clf = GaussianNB()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (30, 2), random_state = 1)
    # clf = tree.DecisionTreeClassifier()
    # clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5) # loss = hinge|modified_huber|log, penalty="elasticnet"|l1|l2
    # clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

    # clf = SupervisedDBNClassification(hidden_layers_structure=[45, 2], learning_rate_rbm=0.05, learning_rate=0.1, n_epochs_rbm=3,
    #                                          n_iter_backprop=100, batch_size=32, activation_function='relu', dropout_p=0.2)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X_resampled, y_resampled):
        x_train, x_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        ### Performance Evaluation Metrics #############
        accuracy = accuracy_score(y_test, y_pred)
        bal_accuracy = balanced_accuracy_score(y_test, y_pred)
        preci_score = precision_score(y_test, y_pred)  # average='macro' or 'micro' or 'weighted'
        recal_score = recall_score(y_test, y_pred)  # average='macro'(default) or 'micro' or 'weighted'
        f_score = f1_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)  # The Kappa statistic varies from 0 to 1
        # compute AUC metric for each loop CV fold
        probs = clf.predict_proba(x_test)[::, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
        roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'
        # roc_auc = roc_auc_score(y_test, probs)
        print("AUC (fold): %f" % (roc_auc))
        # print('Cohens kappa: %f' % kappa)
        #####################################################

        mean_auc.append(roc_auc)
        accuracy_scores.append(accuracy)
        bal_acc.append(bal_accuracy)
        precision_lst.append(preci_score)
        recall_lst.append(recal_score)
        f1_lst.append(f_score)
        kappa_lst.append(kappa)
        fpr_lst.append(fpr)
        tpr_lst.append(tpr)

    print('\n')
    print("Mean AUC: %f" % np.mean(mean_auc))
    print('Mean Accuracy', np.mean(accuracy_scores))
    print('Mean Balanced Accuracy', np.mean(bal_acc))
    print("Mean Precision: %f" % np.mean(precision_lst))
    print('Mean Recall', np.mean(recall_lst))
    print('Mean F_measure', np.mean(f1_lst))
    print('Mean Kappa', np.mean(kappa_lst))

    ########### code computes relevance score of features when classifier is RANDOMfOREST ######################
    # names = list(ds.columns.values)[:]
    # print(names)
    # print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True))
    ##########################################

    # Fast method for cross validation train_test
    # scores = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=10)

    ############### Plot traning and test data ###################
    # plt.scatter(y_test, y_pred)
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    #
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ###################################################
    ############### Plot ROC curve ###################
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % np.mean(mean_auc))
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
    # ###################################################
    ############### Plot precision-recall chart ###################
    from inspect import signature
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


def test_validate(ds):
    # function runs validation data on a trained model
    y_test = ds.pop('label')
    # index = ds.pop('FlybaseId')
    ds = ds.iloc[:, 1:]
    # print(ds.head())
    print(y_test.value_counts())
    x_test = preprocessing.scale(ds)

    # load classifier from pickle object
    print('loading classifier from pickle file...')
    path = 'output/fly/classifier/codon_TMHMM_physico_func_subloc_RNA_homo_PPI.pkl'
    file_obj = open(path, 'rb')
    clf = pickle.load(file_obj)
    file_obj.close()

    print('Fit classifier and evaluate performance...')
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    preci_score = precision_score(y_test, y_pred, average='weighted')  # average='macro' or 'micro' or 'weighted'
    recal_score = recall_score(y_test, y_pred, average='weighted')  # average='macro'(default) or 'micro' or 'weighted'
    f_score = f1_score(y_test, y_pred, average='weighted')

    # compute AUC metric for each loop CV fold
    probs = clf.predict_proba(x_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
    roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'

    print("AUC (fold): %f" % (roc_auc))
    print('Mean Accuracy', accuracy)
    print('Mean Balanced Accuracy', bal_accuracy)
    print("Mean Precision: %f" % preci_score)
    print('Mean Recall', recal_score)
    print('Mean F_measure', f_score)
    print(classification_report(y_test, y_pred))
    ############### Plot ROC curve ###################
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    ###################################################

def test_predict(ds):
    # function runs validation data on a trained model
    ds.dropna(inplace=True)
    index = ds.pop('FlybaseId')

    # print(ds.head())
    # print(y_test.value_counts())
    x_test = preprocessing.scale(ds)

    # load classifier from pickle object
    print('loading classifier from pickle file...')
    path = 'output/fly/classifier/codon_TMHMM_physico_func_subloc_RNA_homo_PPI.pkl'
    file_obj = open(path, 'rb')
    clf = pickle.load(file_obj)
    file_obj.close()

    print('Fit classifier and evaluate performance...')
    y_pred = clf.predict(x_test)

    outFile = pd.DataFrame.from_records([index, y_pred])
    outFile = outFile.transpose()
    outFile.columns = ['FlybaseId','predicted_label']
    outFile.to_csv('output/fly/data/result.csv', index=None)
    print(outFile['predicted_label'].value_counts())
    print(outFile.head())

def train_test_tuning(ds):
    from imblearn.over_sampling import SMOTE

    # bal_acc, accuracy_scores, mean_auc, precision_lst, recall_lst, f1_lst = [], [], [], [], [], []
    # tpr_lst, fpr_lst = [], []
    y = ds.pop('label')
    print(y.value_counts())
    ds = ds.iloc[:, 1:]
    # print(ds.head())
    X = preprocessing.scale(ds)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    clf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_resampled, y_resampled)
    # print(rf_random.best_params_)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    preci_score = precision_score(y_test, y_pred, average='weighted')  # average='macro' or 'micro' or 'weighted'
    recal_score = recall_score(y_test, y_pred, average='weighted')  # average='macro'(default) or 'micro' or 'weighted'
    f_score = f1_score(y_test, y_pred, average='weighted')

    # compute AUC metric for each loop CV fold
    # probs = clf.predict_proba(X_test)[::, 1]
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
    # roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'

    # print("AUC (fold): %f" % (roc_auc))
    print('Mean Accuracy', accuracy)
    print('Mean Balanced Accuracy', bal_accuracy)
    print("Mean Precision: %f" % preci_score)
    print('Mean Recall', recal_score)
    print('Mean F_measure', f_score)

def train_test_tuning_gridsearch(ds):
    from imblearn.over_sampling import SMOTE

    bal_acc, accuracy_scores, mean_auc, precision_lst, recall_lst, f1_lst = [], [], [], [], [], []
    # tpr_lst, fpr_lst = [], []
    y = ds.pop('label')
    print(y.value_counts())
    ds = ds.iloc[:, 1:]
    # print(ds.head())
    X = preprocessing.scale(ds)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    clf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    # Fit the random search model
    grid_search.fit(X_resampled, y_resampled)
    # print(grid_search.best_params_)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    preci_score = precision_score(y_test, y_pred, average='weighted')  # average='macro' or 'micro' or 'weighted'
    recal_score = recall_score(y_test, y_pred, average='weighted')  # average='macro'(default) or 'micro' or 'weighted'
    f_score = f1_score(y_test, y_pred, average='weighted')

    # compute AUC metric for each loop CV fold
    probs = clf.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
    roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'

    print("AUC (fold): %f" % (roc_auc))
    print('Mean Accuracy', accuracy)
    print('Mean Balanced Accuracy', bal_accuracy)
    print("Mean Precision: %f" % preci_score)
    print('Mean Recall', recal_score)
    print('Mean F_measure', f_score)

def DBN_model(ds):
    from dbn import SupervisedDBNClassification
    # ds.fillna(-99999, inplace=True)
    y = ds.pop('label')
    print(y.value_counts())
    X = ds.iloc[:,1:]
    # # print(list(X))
    print(X.shape)
    # Data scaling
    X = preprocessing.scale(X)
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
    classifier = SupervisedDBNClassification(hidden_layers_structure=[800],
                                             learning_rate_rbm=0.00005,   # Default: 0.05
                                             learning_rate=0.001,       # Default: 0.1
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.fit(X_resampled, y_resampled)

    # # Save the model
    # classifier.save('output/pickles/DBN_model_resampled.pkl')
    #
    # # Restore it
    # classifier = SupervisedDBNClassification.load('output/pickles/DBN_model_resampled.pkl')
    # Test
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Done.\nAccuracy: %f' % accuracy_score(y_test, y_pred))
    probs = classifier.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)  # I replaced y_pred with probs
    roc_auc = metrics.auc(fpr, tpr)  # average='macro'(default) or 'micro'
    #
    print("AUC (fold): %f" % (roc_auc))

def calculate_optimal_features_total(ds):
    from sklearn.feature_selection import RFECV
    from imblearn.over_sampling import SMOTE
    # print(sorted(metrics.SCORERS.keys()))
    # ds.fillna(-99999, inplace=True)
    y = ds.pop('label')
    # ds = ds.iloc[:, 1:]
    ds = ds.iloc[:,1:]
    # print(ds.head())
    no_features = len(list(ds))
    X = preprocessing.scale(ds)

    X_resampled, y_resampled = SMOTE().fit_resample(X, y)

    # Recursive feature elimination with cross-validation
    # Create the RFE object and compute a cross-validated score.
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='roc_auc')
    rfecv.fit(X_resampled, y_resampled)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Total number of features : %d" % no_features)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

def feature_selection_auto(dataset):
    # Didnt work
    from feature_selector import FeatureSelector
    # Total features =  61

    y = dataset.pop('label')
    X = dataset.iloc[:, 2:]

    # Features are in train and labels are in train_labels
    fs = FeatureSelector(data=X, labels=y)
    fs.identify_all(selection_params={'missing_threshold': 0.9,
                                      'correlation_threshold': 0.8,
                                      'task': 'classification',
                                      'eval_metric': 'auc',
                                      'cumulative_importance': 0.99})
    # dataframe of collinear features
    print(fs.record_collinear())
    # plot all the correlations
    # fs.plot_collinear(plot_all=True)
    # list of zero importance features
    print(fs.ops['zero_importance'])
    # plot the feature importances
    fs.plot_feature_importances(threshold=0.99, plot_n=12)
    # view all the feature importances in a dataframe
    print(fs.feature_importances)
    # find any columns that have a single unique value
    print(fs.identify_single_unique())
    # Remove the features from all methods (returns a df)
    train_removed_all = fs.remove(methods='all', keep_one_hot=False)
    print(train_removed_all)


#####################################
# Code shows relevant features based on the number of optimal features earlier computed
#####################################
# RFE() requires the number optimal of optimal features computed by feature_selection_RFECV()
# dataset is the data with all features
def feature_selection_RFE(dataset):
    from sklearn.feature_selection import RFECV
    from sklearn.feature_selection import RFE
    from imblearn.over_sampling import SMOTE

    y = dataset.pop('label')
    ds = dataset.iloc[:, 1:]
    cols = list(ds)
    print('Preprocessing Data and constructing Model...')

    # convert values to float
    numpyMatrix = ds.values.astype(float)
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    X = scaler.fit_transform(numpyMatrix)

    X_resampled, y_resampled = SMOTE().fit_resample(X, y)

    # Recursive feature elimination with cross-validation
    # Create the RFE object and compute a cross-validated score.
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='roc_auc')
    rfecv.fit(X_resampled, y_resampled)
    #
    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Total number of features : %d" % len(cols))

    # ncols = rfecv.n_features_       # total no of optimal features
    rfe = RFE(model, 1)
    fit = rfe.fit(X_resampled, y_resampled)
    # fit = rfe.fit(X,y)

    # summarize the selection of the attributes
    temp = pd.Series(rfe.support_, index=cols)
    selected_features_rfe = temp[temp == True].index
    # ss = rfecv.grid_scores_
    print(selected_features_rfe)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    # print("Grid Scores %s" % ss)
    # plt.plot(range(len(ss)), ss)  # try this line

    # lst = rfecv.get_support()
    # indices = find(lst, True)
    # return X[:, indices]
    # print(len(selected_features_rfe))

    f = pd.DataFrame([cols, fit.ranking_.tolist()])
    f = f.transpose()
    f.to_csv('input/fly/testdata/feature_rank_thomas_GOterms.csv', index=None)
    print(f.head())
    # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()

def feature_selection_LASSO(ds):
    y = ds.pop('label')
    ds = ds.iloc[:, 1:]
    # print(ds.head())
    cols = list(ds)
    X = preprocessing.scale(ds)

    reg = LassoCV(cv=5,max_iter=10000)
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X, y))
    coef = pd.Series(reg.coef_, index=cols)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")

    imp_coef = coef.sort_values()
    print(coef[coef == 0])
     # Plot feature importance chart
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")
    plt.show()

def data_imbalance(ds):
    from imblearn.over_sampling import SMOTE, ADASYN
    from collections import Counter
    import pickle

    y = ds.pop('label')
    ds = ds.iloc[:, 1:]
    X = preprocessing.scale(ds)
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    data = np.concatenate((y_resampled[:,None],X_resampled), axis=1)
    data = pd.DataFrame(data)

    # print(data.info())
    # get the count of each class
    print(data[0].value_counts())
    # print(sorted(Counter(y_resampled).items()))
    #
    file_obj = open('input/resampled/features_55.csv', 'wb')
    pickle.dump(data, file_obj)


    # Transform into binary classification
    # ds['label'] = [1 if b == 'B' else 0 for b in ds.label]

    # check the class that the model is biased towards

    # print(np.unique(pred_y))
    # sm = ADASYN()
    # X, y = sm.fit_sample(X, y)

def cor_coeff(ds):
    # import pickle, pandas
    # data = ds
    data = ds.iloc[:,2:]
    # print(data.head())
    corr = data.corr()
    path = 'output/fly/feature_relevance.csv'
    # file_obj = open(path, 'wb')
    # pickle.dump(corr, file_obj)
    # file_obj.close()

    # corr.to_csv(path)
    #
    # Correlation with output variable
    cor_target = abs(corr["label"])
    relevant_features = cor_target[cor_target > 0.001]
    file_obj = pd.DataFrame(relevant_features)
    file_obj.to_csv(path)
    print(relevant_features)


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    # fig.colorbar(cax)
    # ticks = np.arange(0, len(data.columns), 1)
    # ax.set_xticks(ticks)
    # plt.xticks(rotation=90)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(data.columns)
    # ax.set_yticklabels(data.columns)
    # plt.show()

    # plot each category of features separately
    # pd.plotting.scatter_matrix(data.iloc[:,2:10], figsize=(10, 10))
    # plt.show()
    print('Completed Successfully!')

def remove_redundant_features(path, useless_features, coef = 0.7):
    # Code selects non-redundant features based on corr of the features with class label
    # feature rank file contains ranked list of features based on corr with class label
    # coef is the threshold to label 2 features as highly correlated
    # import pandas as pd, pickle
    from collections import OrderedDict

    # load features to exclude from file
    with open(useless_features, 'r') as excl:
        exclude_cols = [x.strip('\n') for x in excl]
    # print(exclude_cols)

    ds = pd.read_csv(path, usecols=lambda x: x not in exclude_cols)
    data = ds.iloc[:, 1:]
    all_features = list(data) # Get list of data features
    # print(len(all_features))
    # exit()
    corr = data.corr()      # Compute correlation coefficient of all features
    cor_target = abs(corr["label"]) # Extract corr coef of class label
    relevant_features = cor_target[cor_target > 0.01] # Extract all features with corr coef > 0.001

    # rank_file = pd.read_csv('output/feature_target_relevance_NoFunctional.csv', header=None)
    rank_file = pd.DataFrame(relevant_features)
    # rank_file.to_csv('input/fly/rankfile2.csv')
    # exit()
    rank_file = rank_file[1:]
    rank_file.sort_values(by=['label'], ascending=False, inplace=True)
    # print(rank_file)

    rank_dict = {}
    count = 0
    optimal_features = set()
    # convert the csv file to a dictionay of features:rank
    for index, item in rank_file.iterrows():
        count +=1
        rank_dict[index] = count
    # print(rank_dict)
    rank_ordered_dict = OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1], reverse= True))
    # print(rank_ordered_dict)
    # select features without corr > abs(0.7) with other features
    # select feature with highest rank if it correlates with multiple features
    keys = [x[0] for x in rank_ordered_dict.items()]
    # print(keys)
    for key in keys:
        key_others = keys.copy()
        key_others.remove(key) # Ensure the remove is local
        best_feature = key
        for other in key_others:
            if abs(corr[key][other]) > coef:
                if rank_dict[key] > rank_dict[other]:
                    # remove key from optimal_features if it already exists
                    optimal_features.remove(key) if key in optimal_features else None
                    if rank_dict[best_feature] > rank_dict[other]:
                        optimal_features.remove(best_feature) if best_feature in optimal_features else None
                        best_feature = other
                else: continue
            else: continue
        optimal_features.add(best_feature)

    with open('input/fly/testdata/optimal_features.txt', 'w') as optimal:
        optimal.write('FlybaseId' + '\n' + 'label' + '\n')
        for item in optimal_features: optimal.write(str(item) + '\n')

    print('Total No of features is %d' % len(all_features))
    print('No of Optimal features is %d' % len(optimal_features))
    print(optimal_features)
    print(set(all_features).difference(set(optimal_features)))

    ########### Correlation Plot ######################
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

#####################################
# functions resamples data to increase obs
#####################################
def resample_data(ds):
    # smooth bootstrapping method
    # Function accepts a small sized dataset and systematically increases the size
    from sklearn.utils import resample
    # from sklearn.datasets import make_classification
    # import numpy as np

    ######### Generate synthetic dataset #####################
    # X, y = make_classification(
    #     n_classes=2, class_sep=1.5, weights=[0.5, 0.5],
    #     n_informative=3, n_redundant=1, flip_y=0,
    #     n_features=5, n_clusters_per_class=1,
    #     n_samples=5, random_state=10
    # )
    # df = pd.DataFrame(X)
    # df['target'] = y
    ##########################################

    ds['label'] = ds['label'].astype('int64')
    y = ds.pop('label')
    df = ds.loc[:, 'T3s':]
    # X = preprocessing.scale(ds)
    # df = pd.DataFrame(X)
    df['target'] =  y
    dim = df.shape
    # creating a noise with the same dimension as the dataset (2,2)
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, dim)

    # ds_resample = pd.DataFrame()
    for i in range(1,20):
        ds_new = resample(df, n_samples=204, random_state=0)
        target = ds_new.iloc[:,-1]
        ds_new = ds_new + noise
        ds_new['target'] = target
        df = df.append(ds_new)
    df.to_csv('output/resampled_data.csv')
    print('Completed')
###################### End of Class count #####################

#####################################
# functions counts freq of label class
#####################################
def class_count():
    # run test codes
    # import pandas as pd
    df = pd.read_csv('output/all_features/all_sequence_homology_rnaSeq_unsigned_functional_PPItopology.csv', usecols=['FlybaseId', 'label'])
    res = df.groupby('label').size()
    # print(df.isnull().sum()) # output no of null values
    print(df.count())      # no of items in the dataframe
    print(res)             # class distribution

    # Alternative method to get class count
    # df['balance'].value_counts()
###################### End of Class count #####################


#####################################
# # Discretization functions converts continuous variables to
# multiple binary variables depending on the bucket size
#####################################
# takes multiple continuous feature and discretize it
def discretization(ds, cols):
    # ds_new = ds.iloc[:,:2]
    # others = ds.iloc[:,2:16] # adjust the value according to position of interested features
    # create a new ds to store new features
    data = ds.iloc[:,:3]
    cols = cols[3:] # select all features from column name of input data
    # print(data.head())
    # discretize all continuous features and add new binary features to existing binary
    # features
    suffix = [x for x in range(20)]
    # divides values in each column to a bucket of 20
    for col in cols:
        lst = ds.pop(col)
        res = pd.qcut(lst, 20, duplicates='drop', labels=False)
        for item in suffix:
            col_new = col + '_' + str(item)
            data[col_new] = [1 if i == item else 0 for i in res]

    # merge all data
    # ds_new = ds_new.join(others)
    # ds_new = ds_new.join(data)
    data.to_csv('output/fly/data/optimized_discretized.csv', index=None)
    print(data.head())

# takes a continuous feature and discretize it
# manual means it smaller buckets are subsets of superior buckets
def discretization_manual(ds, cols, f_name, n=20):
    # n is the number of buckets to divide the variable
    # np.random.seed(1)
    # read in input
    # inp = np.random.uniform(range(0, 20))
    outpath = 'output/beetle/data/discretize_' + f_name + '_NoRedun.csv'
    cal = len(list(ds))-2
    print('Total input features is %d' % cal)
    data = ds.iloc[:, :2]

    # cols = list(ds)
    cols = cols[2:]  # select all features from column name of input data
    print('Looping through features to generate buckets...')
    for col in cols:
        inp = ds.pop(col)
        a = min(inp)
        b = max(inp)
        # print(inp)
        # divide the diff between max and min into n parts (buckets)
        buck = np.linspace(a, b, n+1)
        # print(buck)
        # assign each obs into a bucket if inclusion criteria is met equal parts
        sup = len(buck) - 1
        for index, loc in enumerate(buck):
            if index < sup:
                id = col + '_' + str(index)
                data[id] = [1 if x >= loc else 0 for x in inp]

    print('Total output features is %d' % len(list(data)))
    data.to_csv(outpath, index=None)
############################## End of Discretization functions ####################################

if __name__ == "__main__":
    # file_name = 'codon_TMHMM_physico_func_subloc_RNA_homo_PPI'  # fly
    file_name = 'GO_15cutoff_thomas'
    # file_name = 'discretize_codon_TMHMM_homo_subloc_rna_physico_func_PPI_NoRedun'  # beetle
    path = 'output/fly/data/'+ file_name + '.csv'

    # path = 'input/fly/testdata/' + file_name + '.csv'
    # path = 'output/beetle/data/codon_TMHMM_homo_subloc_rna_physico_func_PPI.csv'

    # ds = pd.read_csv(path, usecols=lambda x: x not in columns_to_skip)
    ds = pd.read_csv(path)
    # ds = pd.read_csv(path, usecols=optimal_cols)
    ############ Core Functions ##################
    # Data cleaning must be done first. See manipulate_sequence script for the function
    feature_selection_RFE(ds)   # 1 : Determines number of Optimal features
    # useless_features = 'input/fly/testdata/excluded_features.txt'
    # optimal_features = 'input/fly/testdata/optimal_features.txt'
    # remove_redundant_features(path, useless_features) # 2: Computes correlation score of all features with respect to the label and use that to Remove redundant features
    # train_test(path,optimal_features, file_name)                # 3: Train model
    # train_test(path)
    # test_validate(ds)             # 4: Validate model (Upsampling after splitting train/test data)
    # test_predict(ds)
    # DBN_model(ds)
    # feature_selection_RFE44()
    ############ Utility Functions ##################
    # resample_data(ds)
    # data_imbalance(ds)
    # class_count()
    # discretization_manual(ds, optimal_cols, file_name)

    ########## Other Functions ######################
    # feature_selection_auto(ds)

    # discretization_bulk(ds, cols)    #
    # calculate_optimal_features_total(ds)
    # feature_selection_LASSO(ds)         # Determine optimal features which is used as param for reading input data
    # train_test_simple(ds, test_data)    # Downsampling before splitting train/test data
    # train_test_basic(ds)              # Upsampling before splitting train/test data
    # train_test_beta(ds, test_data)
    # train_test_tuning(ds)
    # print(len(list(ds)))
# Handle the issue of class imbalance
# Perform either PCA or feature selection to prevent overfitting

### Notes ###
# 1. After data cleaning, run feature_selection_RFE(ds) to get useless features
# 2. Extract useless features from 1 and pass path to remove_redundant_features(path) to obtain optimal features
# 3. Run train_test(path,optimal_features, file_name) to train model