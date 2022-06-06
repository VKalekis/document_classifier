from lsh_modules import get_forest, queryForest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.feature_extraction.text import CountVectorizer
import math
import warnings
from datetime import datetime

"""
    Run KNN-LSH classification method on the unknown test set on Kaggle.
"""


def knn_func(X_train, y_train, X_test):
    """
    Run knn function classifier (using sk.learn.KNeighborsClassifier), fitted on train set, and use it to make predictions on the test set.
    Uses k=15 neighbors if len(X_train)>15, otherwise uses k=sqrt(len(X_train)).
    """

    if len(X_train) < 225:
        n_neighbors = math.ceil(math.sqrt(len(X_train)))
    else:
        n_neighbors = 15

    predictions = []

    start1 = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm="brute", metric="jaccard", n_jobs=1
    )

    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    knntime = time.time() - start1

    return predictions, 0, knntime


def knn_lsh(train_df, test_df, perms, threshold):
    """
    Implements knn-LSH classifier with LSH as a preprocessing step.
    For each test document, we query the LSH forest.
        If no results are obtained, we run knn on the entire train dataset.
        If the LSH returns results for the test document, we run the knn using those results as the train set of the classifer.
    """

    # Keep reviews column.
    X_train = train_df.iloc[:, 1].values
    # Keep sentiment column.
    y_train = train_df.iloc[:, 2].values

    # Keep reviews column.
    X_test = test_df.iloc[:, 1].values

    # Create forest of MinHashed values from Train dataset.
    forest, forest_time = get_forest(X_train, perms, threshold)
    # Query previously created forest with values from Test dataset.
    results, queries_time = queryForest(X_test, forest, perms)

    # Initialize and create vectorizer for train and test dataset.
    # If memory allocation error, use .astype("int8") after transform command.
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    start_time = time.time()

    zero_results_indexes = []
    nonzero_results_indexes = []
    nonzero_results = []

    # Decouple test documents that the query returned as similar ones, from the ones that did not find any similarities.
    for index, result in enumerate(results):
        if len(result) == 0:
            zero_results_indexes.append(index)
        else:
            nonzero_results_indexes.append(index)
            nonzero_results.append(result)

    print(f"LSHED{len(nonzero_results_indexes)}")

    # For those that no similar documents were found, do KNN by using the whole train dataset.
    if len(zero_results_indexes) != 0:
        bf_preds, _, _ = knn_func(
            X_train, y_train, np.take(X_test, zero_results_indexes, axis=0)
        )
    else:
        bf_preds = []

    # Create a subset with the similar documents that were found and do KNN with this subset as the train one.
    lsh_preds = []

    for index, item in enumerate(nonzero_results_indexes):

        result = nonzero_results[index]

        X_train_t = np.take(X_train, result, axis=0)
        y_train_t = np.take(y_train, result, axis=0)

        test = np.take(X_test, item, axis=0)
        reshaped_test = test.reshape(1, -1)

        # Perform KNN on each test document using its results from the LSH as its train set.
        pred, _, _ = knn_func(X_train_t, y_train_t, reshaped_test)

        # print(pred)
        lsh_preds.append(pred[0])

    # FIXED TOTAL_PREDS
    total_preds = np.zeros(len(test_df))

    # print(len(zero_results_indexes))
    # print(len(bf_preds))
    # print(len(nonzero_results_indexes))
    # print(len(lsh_preds))

    total_preds[zero_results_indexes] = bf_preds
    total_preds[nonzero_results_indexes] = lsh_preds

    lsh_time = time.time() - start_time

    total_time = forest_time + queries_time + lsh_time

    return total_preds, total_time


def knn_bf(train_df, test_df):
    """
    Do KNN with the brute force methods by providing the entire train and test dataset.
    """
    X_train = train_df.iloc[:, 1].values
    y_train = train_df.iloc[:, 2].values

    X_test = test_df.iloc[:, 1].values

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    start_time = time.time()

    predictions_bf, _, _ = knn_func(X_train, y_train, X_test)

    bf_time = time.time() - start_time

    return predictions_bf, bf_time


warnings.filterwarnings("ignore")

print(datetime.now())

# Read train dataset
train_df = pd.read_csv(r"PATH_OF_TRAIN_DATASET_GOES_HERE", encoding="utf-8")

# Read test dataset
test_df = pd.read_csv(r"PATH_OF_TEST_DATASET_GOES_HERE", encoding="utf-8")
# Print lengths.
print(len(train_df))
print(len(test_df))

# Keep id column
test_id = test_df["id"]


# Uncomment the following section to implement KNN by using the Brute Force method.

# bf_preds, bf_time = knn_bf(train_df, test_df)

# df = pd.DataFrame(list(zip(test_id, bf_preds)),
#                columns =['id', 'sentiment'])
# df.to_csv(f'/home/vkalekis/projects/bigdata/src2/bf_resultsontestset.csv',index=False)


# Setting up number of permutation and threshold values.
perms = 64
# thrsld = 0.3
thrsld_list = [0.35, 0.4, 0.5, 0.6, 0.7, 0.8]

for thrsld in thrsld_list:
    # Call KNN+LSH pipeline.
    lsh_preds, lsh_time = knn_lsh(train_df, test_df, perms, thrsld)

    print(thrsld)
    print(lsh_time)

    # Save calculations to Dataframe.
    df = pd.DataFrame(
        list(zip(test_id, lsh_preds.astype("int"))), columns=["id", "sentiment"]
    )

    # Save Dataframe to .csv file.
    df.to_csv(f"OUTPUT_PATH_GOES_HERE", index=False)
    print(datetime.now())
