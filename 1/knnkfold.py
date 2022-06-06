"""
    Runs the knn classifer on the preprocessed dataset using k folds to minimize the bias from choosing only one training and test set.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, neighbors
from sklearn.feature_extraction.text import CountVectorizer
import time
from datetime import datetime
from sklearn.model_selection import KFold

len_train_and_test = 500

df = pd.read_csv(
    r"/home/vkalekis/Documents/bigdata/dataframes/Train_Preprocessed_new.csv",
    encoding="utf-8",
    nrows=len_train_and_test,
).fillna(" ")
print(df.head)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
k = 0

neighbors = []
accs = []
macrof1s = []
times = []

for train_indexes, test_indexes in kf.split(df):
    print(k)

    # Extract train and test sets from dataframe.
    train_df = df.iloc[train_indexes, :]
    test_df = df.iloc[test_indexes, :]

    X_train = train_df.iloc[:, -2].values
    y_train = train_df.iloc[:, -1].values

    X_test = test_df.iloc[:, -2].values
    y_test = test_df.iloc[:, -1].values

    # Vectorization using CountVectorizer.
    vectorizer = CountVectorizer(max_features=60000)

    X_train = vectorizer.fit_transform(X_train).astype("int8").toarray()
    X_test = vectorizer.transform(X_test).astype("int8").toarray()

    # Run knn classifer on all folds using all the candidate neighbor values.
    # Then, collect data for accuracies, macrof1s and times and save them to a .csv for plotting
    # and further inspection.

    candidate_neighbors = [1, 3, 5, 7, 9, 15, int(np.sqrt(len(train_df)))]

    for neighbor in candidate_neighbors:

        t1 = time.time()

        knn = KNeighborsClassifier(n_jobs=6, n_neighbors=neighbor, metric="jaccard")

        knn.fit(X_train, y_train)
        predicted = knn.predict(X_test)

        acc = metrics.accuracy_score(y_test, predicted)
        macrof1 = metrics.f1_score(y_test, predicted, average="macro")

        t2 = time.time()

        neighbors.append(neighbor)
        accs.append(acc)
        macrof1s.append(macrof1)
        times.append((t2 - t1) / 60.0)

    k += 1

results_df = pd.DataFrame(
    list(zip(neighbors, accs, macrof1s, times)),
    columns=["neighbors", "accs", "macrof1s", "times"],
)

results_df.to_csv(
    r"/home/vkalekis/Documents/bigdata/deliverables/results.csv", index=False
)
