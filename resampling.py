#Imbalanced datasets
#Source: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets/notebook
import pandas as pd
def target_proportion():
    import pandas as pd

    df_train = pd.read_csv('../input/train.csv')

    target_count = df_train.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

    target_count.plot(kind='bar', title='Count (target)');

def accuracy_score_evaluation():
    ###   One of the major issues that novice users fall into when dealing with
    # unbalanced datasets relates to the metrics used to evaluate their model.
    # Using simpler metrics like accuracy_score can be misleading.
    # In a dataset with highly unbalanced classes, if the classifier always "predicts"
    # the most common class without performing any analysis of the features,
    # it will still have a high accuracy rate, obviously illusory.
    ###
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df_train = pd.read_csv('../input/train.csv')
    # Remove 'id' and 'target' columns
    labels = df_train.columns[2:]

    X = df_train[labels]
    y = df_train['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    #Now let's run the same code, but using only one feature (which should drastically reduce the accuracy of the classifier):
    model = XGBClassifier()
    model.fit(X_train[['ps_calc_01']], y_train)
    y_pred = model.predict(X_test[['ps_calc_01']])

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

def confusion_matrix(y_test, y_pred):
    ### An interesting way to evaluate the results is by means of a confusion matrix,
    # which shows the correct and incorrect predictions for each class.
    ###
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('Confusion matrix:\n', conf_mat)

    labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()

def random_under_sampling():
    ### uses the DataFrame.sample method to get random samples each class ###

    df_train = pd.read_csv('../input/train.csv')
    # Class count
    count_class_0, count_class_1 = df_train.target.value_counts()

    # Divide by class
    df_class_0 = df_train[df_train['target'] == 0]
    df_class_1 = df_train[df_train['target'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1)
    df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    print('Random under-sampling:')
    print(df_test_under.target.value_counts())

    df_test_under.target.value_counts().plot(kind='bar', title='Count (target)');


def random_over_sampling():
    df_train = pd.read_csv('../input/train.csv')
    # Class count
    count_class_0, count_class_1 = df_train.target.value_counts()

    # Divide by class
    df_class_0 = df_train[df_train['target'] == 0]
    df_class_1 = df_train[df_train['target'] == 1]

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    print('Random over-sampling:')
    print(df_test_over.target.value_counts())

    df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');

def sample_dataset_unbalanced():
    ### For ease of visualization, let's create a small unbalanced
    # sample dataset using the make_classification method:
    ###
    import imblearn
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
        n_informative=3, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1,
        n_samples=100, random_state=10
    )

    df = pd.DataFrame(X)
    df['target'] = y
    df.target.value_counts().plot(kind='bar', title='Count (target)');


def plot_2d_space(X, y, label='Classes'):
    # We will also create a 2-dimensional plot function, plot_2d_space, to see the data distribution:
    from matplotlib import pyplot as plt
    import numpy as np

    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

def plot_sample_dataset_unbalanced():
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

def undersampling_with_imbalanced():
    #Random under-sampling and over-sampling with imbalanced-learn
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(return_indices=True)
    X_rus, y_rus, id_rus = rus.fit_sample(X, y)

    print('Removed indexes:', id_rus)

    plot_2d_space(X_rus, y_rus, 'Random under-sampling')

def oversampling_with_imbalanced():
    #Random over-sampling and over-sampling with imbalanced-learn
    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(X, y)

    print(X_ros.shape[0] - X.shape[0], 'new random picked points')

    plot_2d_space(X_ros, y_ros, 'Random over-sampling')

def undersampling_tomlinks():
    ### Tomek links are pairs of very close instances, but of opposite classes.
    # Removing the instances of the majority class of each pair increases the space
    # between the two classes, facilitating the classification process.
    ###
    # In the code below, we'll use ratio='majority' to resample the majority class.

    from imblearn.under_sampling import TomekLinks

    tl = TomekLinks(return_indices=True, ratio='majority')
    X_tl, y_tl, id_tl = tl.fit_sample(X, y)

    print('Removed indexes:', id_tl)

    plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')

def undersampling_cluster_centroids():
    ### This technique performs under-sampling by generating centroids based on clustering methods.
    # The data will be previously grouped by similarity, in order to preserve information.
    #In this example we will pass the {0: 10} dict for the parameter ratio,
    # to preserve 10 elements from the majority class (0), and all minority class (1) .
    ###
    from imblearn.under_sampling import ClusterCentroids

    cc = ClusterCentroids(ratio={0: 10})
    X_cc, y_cc = cc.fit_sample(X, y)

    plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')

def undersampling_cluster_centroids():
    ### SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing
    # elements for the minority class, based on those that already exist.
    # We'll use ratio='minority' to resample the minority class.
    ###
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)

    plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')

def over_sampling_under_sampling():
    ### Now, we will do a combination of over-sampling and under-sampling,
    # using the SMOTE and Tomek links techniques:###
    from imblearn.combine import SMOTETomek

    smt = SMOTETomek(ratio='auto')
    X_smt, y_smt = smt.fit_sample(X, y)

    plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')