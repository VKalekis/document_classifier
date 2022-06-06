from lsh_modules import get_forest, queryForest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import math
import warnings
from datetime import datetime

"""
    This file implements a knn+LSH classifier using a Local Vectorizer.
    As Global Vectorizer we define a CountVectorizer which vectorizes the documents in a global scope,
    in contrast to the Local Vectorizer, which vectorizes the documents at a local scope, before each knn classification.
"""


def knn_func(X_train, y_train, X_test):
    """
    Run knn function classifier, fitted on train set, and use it to make predictions on the test set.
    Uses k=15 neighbors if len(X_train)>15, otherwise uses k=sqrt(len(X_train)).
    """

    if len(X_train) < 225:
        n_neighbors = math.ceil(math.sqrt(len(X_train)))
    else:
        n_neighbors = 15

    predictions = []

    start1 = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm="brute", metric="jaccard", n_jobs=6
    )

    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    knntime = time.time() - start1

    return predictions, 0, knntime


def knn_lsh(train_df, test_df, perms, threshold):
    """
    Run knn using LSH as a preprocessing step.
    For each test document, we query the LSH forest.
        If no results are obtained, we run knn on the entire train dataset.
        If the LSH returns results for the test document, we run the knn using those results as the train set of the classifer.
    """

    ## Keep reviews column.
    X_train = train_df.iloc[:, 1].values
    # Keep sentiment column.
    y_train = train_df.iloc[:, 2].values

    # Keep reviews column.
    X_test = test_df.iloc[:, 1].values
    # Keep sentiment column.
    y_test = test_df.iloc[:, 2].values

    # Create forest of MinHashed values from Train dataset.
    forest, forest_time = get_forest(X_train, perms, threshold)
    # Query previously created forest with values from Test dataset.
    results, queries_time = queryForest(X_test, forest, perms)

    start_time = time.time()

    # Initialize and create vectorizer for train and test dataset.
    # If memory allocation error, use .astype("int8") after transform command.
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    universal_vectorizations = time.time() - start_time
    print(f"Universal vectorization took {universal_vectorizations}")

    # Time for different portions of the execution.
    start_time = time.time()
    bf_knn = 0
    lsh_knn = 0
    vectorization_totaltime = 0
    knn_totaltime = 0

    # Decouple test documents that the query returned as similar ones, from the ones that did not find any similarities.
    zero_results_indexes = []
    nonzero_results_indexes = []
    nonzero_results = []

    for index, result in enumerate(results):
        if len(result) == 0:
            zero_results_indexes.append(index)
        else:
            nonzero_results_indexes.append(index)
            nonzero_results.append(result)

    # For those that no similar documents were found, do KNN by using the whole train dataset.
    if len(zero_results_indexes) != 0:
        time1 = time.time()
        bf_preds, _, _ = knn_func(
            X_train, y_train, np.take(X_test, zero_results_indexes, axis=0)
        )
        bf_knn = time.time() - time1
    else:
        bf_preds = []

    # Create a subset with the similar documents that were found and do KNN with this subset as the train one.
    lsh_preds = []
    lsh_labels = []

    for index, item in enumerate(nonzero_results_indexes):

        result = nonzero_results[index]

        X_train_t = np.take(X_train, result, axis=0)
        y_train_t = np.take(y_train, result, axis=0)

        test = np.take(X_test, item, axis=0)
        reshaped_test = test.reshape(1, -1)

        time1 = time.time()

        # Perform KNN on each test document using its results from the LSH as its train set.
        pred, vectime, knntime = knn_func(X_train_t, y_train_t, reshaped_test)

        lsh_knn += time.time() - time1

        vectorization_totaltime += vectime
        knn_totaltime += knntime

        lsh_labels.append(np.take(y_test, item))
        lsh_preds.append(pred[0])

    # print(f"LSH KNN ended {time.time()-temp}")

    # Create vectors for total predictions + labels from BF + LSH.
    total_preds = np.zeros(len(test_df))

    total_preds[zero_results_indexes] = bf_preds
    total_preds[nonzero_results_indexes] = lsh_preds

    total_labels = np.zeros(len(test_df))
    total_labels[zero_results_indexes] = np.take(y_test, zero_results_indexes, axis=0)
    total_labels[nonzero_results_indexes] = np.take(
        y_test, nonzero_results_indexes, axis=0
    )

    # Calculate lsh_accuracy and vector containing the different times.
    lsh_acc = metrics.accuracy_score(total_labels, total_preds)

    lsh_time = time.time() - start_time

    total_time = forest_time + queries_time + lsh_time

    print(f"# of not LSHed: {len(zero_results_indexes)}")
    print(f"BFknns: {bf_knn} LSHknns: {lsh_knn}")
    print(f"Vectorizations: {vectorization_totaltime} Knn: {knn_totaltime}")

    print(f"LSH time {lsh_time}   LSH acc {lsh_acc}")
    print(f"Total time {total_time} ")

    times = [
        forest_time,
        queries_time,
        lsh_time,
        total_time,
        bf_knn,
        lsh_knn,
        universal_vectorizations,
        knn_totaltime,
    ]

    return lsh_acc, times, len(lsh_preds)


def knn_bf(train_df, test_df):
    """
    Do KNN with the brute force methods by providing the entire train and test dataset.
    """

    X_train = train_df.iloc[:, 1].values
    y_train = train_df.iloc[:, 2].values

    X_test = test_df.iloc[:, 1].values
    y_test = test_df.iloc[:, 2].values

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    start_time = time.time()

    predictions_bf, _, _ = knn_func(X_train, y_train, X_test)
    bf_acc = metrics.accuracy_score(y_test, predictions_bf)

    bf_time = time.time() - start_time

    return bf_acc, bf_time


def generateResults(k, outputTable, train_df, test_df):
    """
    Given a train and test dataframe:
        Run BF KNN, recording time + accuracy
        Run LSH KNN, with a set threshold and number of permutations, recording time, accuracy and time metrics.
    Tabulate the results in a .csv file.
    """

    bf_acc, bf_time = knn_bf(train_df, test_df)
    bf_acc, bf_time = 0, 0
    print(f"BFtime {bf_time} BFacc: {bf_acc}")

    outputTable.loc["Brute Force Jaccard_" + str(k)] = {
        "Build Time": 0,
        "Query Time": 0,
        "LSH Time": 0,
        "Total Time": bf_time,
        "BFKNN": 0,
        "LSHKNN": 0,
        "VEC": 0,
        "KNN": 0,
        "# of LSH Preds": 0,
        "Accuracy": bf_acc,
        "Parameters": np.nan,
    }

    threshold_list = np.arange(0.8, 0.2, -0.05)
    perms_list = [16, 32, 64]

    for perms in perms_list:

        for thrsld in threshold_list:

            thrsld = np.around(thrsld, 2)

            print(f"\nParameters: {perms} - {thrsld}")

            lsh_acc, times, len_lsh_preds = knn_lsh(train_df, test_df, perms, thrsld)

            (
                forest_time,
                queries_time,
                lsh_time,
                total_time,
                bf_knn,
                lsh_knn,
                vectorization_totaltime,
                knn_totaltime,
            ) = times

            outputTable.loc[f"LSH Jaccard_{perms}, {thrsld}_{k}"] = {
                "Build Time": forest_time,
                "Query Time": queries_time,
                "LSH Time": lsh_time,
                "Total Time": total_time,
                "BFKNN": bf_knn,
                "LSHKNN": lsh_knn,
                "VEC": vectorization_totaltime,
                "KNN": knn_totaltime,
                "# of LSH Preds": len_lsh_preds,
                "Accuracy": lsh_acc,
                "Parameters": str(perms) + ", " + str(thrsld),
            }
    return outputTable


def ifstatic(df, len_train, len_test):
    """
    For static train and test datasets, take a portion of the head of the dataset as train set and
    from the tail as test set.
    """
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True).fillna(" ")

    train_df = df_shuffled.head(len_train)
    test_df = df_shuffled.tail(len_test)

    return train_df, test_df


warnings.filterwarnings("ignore")

print(datetime.now())

# Size of train set.
len_train = 20000

# Read and shuffle dataset and take len_train samples.
df = pd.read_csv(
    r"/home/vkalekis/projects/bigdata/dfs2/imdb_train_pre.csv", encoding="utf-8"
)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df_shuffled.head(len_train)

# train_df, test_df = ifstatic(df)

# Output table to collect the results.
outputTable = pd.DataFrame(
    columns=[
        "Build Time",
        "Query Time",
        "LSH Time",
        "Total Time",
        "BFKNN",
        "LSHKNN",
        "VEC",
        "KNN",
        "# of LSH Preds",
        "Accuracy",
        "Parameters",
    ]
)

# Run a K-fold validation on the entire dataset, keeping 3/4 as the train set and 1/4 as the test set.
kf = KFold(n_splits=4, shuffle=True, random_state=42)
k = 0
for train_indexes, test_indexes in kf.split(df):
    start_time = time.time()

    print(len(train_indexes))
    print(len(test_indexes))

    # Take the specific rows as the train and test dataset.
    train_df = df.iloc[train_indexes, :]
    test_df = df.iloc[test_indexes, :]

    # Generate output table with the results.
    outputTable = generateResults(k, outputTable, train_df, test_df)

    k += 1
    end_time = time.time() - start_time
    print(end_time)

    # For one k-fold
    # break

# Save outputTable to .csv format.
outputTable.to_csv(
    f"/home/vkalekis/projects/bigdata/src2/{datetime.now()}.outputtable_{len_train}.csv"
)
print(datetime.now())
