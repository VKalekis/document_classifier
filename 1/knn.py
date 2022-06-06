"""
    Runs the knn classification algorithm using Jaccard distance on the preprocessed train dataset. 
    Keeps a portion of the dataset as the train set and the other portion as a test set in order
    to estimate the optimal parameters which hopefully will generalize in a good way to the unknown dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, neighbors
from sklearn.feature_extraction.text import CountVectorizer
import time
from datetime import datetime
import sys


def getXandY(df, len_train, len_test):
    """
    Shuffle df and take:
        len_train samples from beginning as train set.
        len_test samples from end as test set.
    """

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True).fillna(" ")

    train_df = df_shuffled.head(len_train)
    test_df = df_shuffled.tail(len_test)

    X_train = train_df.iloc[:, 0].values
    y_train = train_df.iloc[:, 1].values

    X_test = test_df.iloc[:, 0].values
    y_test = test_df.iloc[:, 1].values

    return X_train, y_train, X_test, y_test


def splitDf(df, split):
    """
    Split df to train/test sets using the train_test_split with a predefined split.
    """
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True).fillna(" ")

    X_train, X_test, y_train, y_test = train_test_split(
        df_shuffled.iloc[:, -2].values,
        df_shuffled.iloc[:, -1].values,
        test_size=split,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


# Command line arguments or default values
if len(sys.argv) > 1:
    len_train = int(sys.argv[1])
    len_test = int(sys.argv[2])
else:
    len_train = 500
    len_test = 100

# Read preprocessed dataset.
print(f"{len_train}   {len_test}")

df = pd.read_csv(
    r"/home/vkalekis/Documents/bigdata/dataframes/Train_Preprocessed_new.csv",
    encoding="utf-8",
    nrows=len_train + len_test,
)

print(df.head)


# X_train, y_train, X_test, y_test = splitDf(df, 0.2)
X_train, y_train, X_test, y_test = getXandY(df, len_train, len_test)

# Marks dataframe as useless for now to be garbage collected.
del df


# Vectorization using CountVectorizer, only keeping the 75k most popular features.
# The int8 representation is used for smaller memory footprint.

vectorizer = CountVectorizer(max_features=75000)

X_train = vectorizer.fit_transform(X_train).astype("int8").toarray()
X_test = vectorizer.transform(X_test).astype("int8").toarray()
print(X_train.shape)


# Run knn neighbors classifier on the datasets with variable number of neighbors.
# The algorithm is fitted on the train set and then predictions are made to the test set.
# The accuracy between the predictions and the true labels is calculated and different metrics (eg executiom time, accuracy, # of neighbors)
# are logged in a .txt file for further inspection.

candidate_neighbors = [5, 9]


for neighbors in candidate_neighbors:

    t1 = time.time()

    knn = KNeighborsClassifier(n_jobs=6, n_neighbors=neighbors, metric="jaccard")

    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)

    acc = metrics.accuracy_score(y_test, predicted)

    macrof1 = metrics.f1_score(y_test, predicted, average="macro")

    t2 = time.time()

    with open(r"/home/vkalekis/Documents/bigdata/deliverables/results.txt", "a") as f:
        f.write(
            f"{datetime.now()} KNN: Train:{len_train} Test:{len_test} Neighbours:{neighbors}  Accuracy:{acc * 100}% Macrof1:{macrof1} Time:{(t2-t1)/60}\n"
        )
