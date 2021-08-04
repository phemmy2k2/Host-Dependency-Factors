import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter


def xgb_selector2(X, y, weight):
    from sklearn.feature_selection import SelectFromModel
    from lightgbm import LGBMClassifier

    # y = df.pop('label')
    # X = df.values
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40, scale_pos_weight=weight) #

    embeded_lgb_selector = SelectFromModel(lgbc) #, max_features=num_feats
    embeded_lgb_selector.fit(X, y)

    embeded_lgb_support = embeded_lgb_selector.get_support()
    return embeded_lgb_support

def preprocess(X):
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer

    X[X == np.inf] = 0
    numpyMatrix = X.astype(np.float64)
    # X = np.nan_to_num(X)

    # replace missing values with mean
    # imputer = SimpleImputer(strategy='median') #missing_values=np.nan, strategy='median'
    # numpyMatrix = imputer.fit_transform(numpyMatrix)

    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    X = scaler.fit_transform(numpyMatrix)
    print('Preprocessing completed!')
    return X

def preprocess_v2(X):
    # similar to preprocess() only has ability to fill in missing values
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer

    X[X == np.inf] = 0
    # numpyMatrix = X.astype(np.float64)
    X = np.nan_to_num(X)

    # replace missing values with mean
    imputer = SimpleImputer(strategy='median') # strategy='median',missing_values=['?', np.nan],
    numpyMatrix = imputer.fit_transform(X)

    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    X = scaler.fit_transform(numpyMatrix)
    print('Preprocessing completed!')
    return X

def run_ml(): # similar to perform_ML3(), compute featImp and confusion matrix foldbyfold added
    print('perform_ML function started...')
    # load in data
    filepath = str(input("Enter path to the labeled data: "))
    df = pd.read_csv(filepath, index_col=0)

    print('Shape of data is %s' % str(df.shape))
    y = df.pop('label')
    # estimate scale_pos_weight value
    counter = Counter(y)
    print(counter)
    estimate = counter[0] / counter[1]
    print(estimate)
    X = preprocess(df.values) # change back to preprocess()

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10) #
    clf = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40, scale_pos_weight=estimate)

    res = {'roc_auc': [], 'roc_pr': [], 'precision': [], 'fscore': [], 'sensi': [], 'speci': [], 'acc': [],'mcc':[]}
    fold_pred = pd.DataFrame()  # stores prediction from each fold
    feat_imp_df = pd.DataFrame()  # stores feature ranking for each fold
    count = 1

    for train_index, test_index in cv.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # perform data resampling to have balanced label class
        X_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
        ind = xgb_selector2(X_resampled, y_resampled, estimate)
        clf.fit(X_resampled[:, ind], y_resampled)

        # predict test data
        y_pred = clf.predict(x_test[:, ind])

        ### Performance Evaluation Metrics #############
        probs = clf.predict_proba(x_test[:, ind])[::, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label=1, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)  # average='macro'(default) or 'micro'
        precision, recall, _ = precision_recall_curve(y_test, probs, pos_label=1)
        pre_recall = auc(recall, precision)

        # get feature ranking for each CV fold
        cols = np.array(df.columns)[ind]
        feat_imp = pd.Series(clf.feature_importances_, name="fold_" + str(count), index=pd.Index(cols, name="feat"))

        # merge featImp for each fold in an output file
        if feat_imp_df.empty:
            feat_imp_df = pd.concat([feat_imp_df, feat_imp], axis=1)
        else:
            feat_imp_df = feat_imp_df.join(feat_imp)

        # get [probs, y_pred and y_true] for each fold
        genes = df.index[test_index]
        pred = {'geneId':genes, 'label':y_test, 'prediction':y_pred,'pred_proba':probs}
        fold = pd.DataFrame(pred)
        fold_pred = pd.concat([fold_pred, fold], axis=0, sort=False)


        cm1 = confusion_matrix(y_test, y_pred)
        total1 = sum(sum(cm1))
        a,b,c,d = cm1[0, 0], cm1[0, 1], cm1[1, 0], cm1[1, 1]
        accuracy = (a + d) / total1
        specificity = a / (a + b)
        sensitivity = d / (c + d)
        mcc = ((d*a)-(b*c)) /np.sqrt((d+b)*(d+c)*(a+b)*(a+c))
        preci_score = precision_score(y_test, y_pred)  # average='macro' or 'micro' or 'weighted'
        f_score = f1_score(y_test, y_pred)

        # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        res['roc_auc'].append(roc_auc)
        res['roc_pr'].append(pre_recall)
        res['precision'].append(preci_score)
        res['fscore'].append(f_score)
        res['sensi'].append(sensitivity)
        res['speci'].append(specificity)
        res['acc'].append(accuracy)
        res['mcc'].append(mcc)
        count += 1

    res_test = {k: np.mean(v) for k, v in res.items()}
    print("AUROC Score: %.3f" % res_test['roc_auc'])
    print("AUPRC Score: %.3f" % res_test['roc_pr'])
    print("Precision Score: %.3f" % res_test['precision'])
    print("F-score: %.3f" % res_test['fscore'])
    print("Sensitivity: %.3f" % res_test['sensi'])
    print("Specificity: %.3f" % res_test['speci'])
    print("Accuracy: %.3f" % res_test['acc'])
    print("MCC: %.3f" % res_test['mcc'])

    # write Evaluation metrics to file
    res = pd.DataFrame.from_dict(res)
    res.to_csv('lgbm_metrics.csv')

    # write predictions to file
    feat_imp_df.to_csv('lgbm_feature_importance.csv', index=False)
    fold_pred.to_csv('lgbm_prediction.csv', index=False)
    print('Completed!')

if __name__ == '__main__':
    run_ml()