"""
    Run knn classifier on the unknown test set for the Kaggle submissions.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import time
from datetime import datetime
import sys

# Command line arguments or default values
if len(sys.argv) > 1:
    len_train = int(sys.argv[1])
else:
    len_train = 40000


# Read dataframe containing the train values.
df = pd.read_csv(
    r"/home/vkalekis/Documents/bigdata/dataframes/Train_Preprocessed.csv",
    encoding="utf-8",
).fillna(" ")


# Extract X and y values.
X_train = df.iloc[:, -2].values
y_train = df.iloc[:, -1].values

# Vectorization.
vectorizer = CountVectorizer(max_features=70000)
X_train = vectorizer.fit_transform(X_train).astype("int8").toarray()

print(X_train.shape)
print(X_train.nbytes / np.power(10, 9))
del df


# Make predictions on the test set using a batch size of 2000 rows for smaller memory footprint.
# At each batch extract the corresponding rows from the test dataset, fit the classifier on the train set and then make predictions on this batch.
# Collect the predictions and then create a results dataframe with the indexes and the predictions.

predictions = []
batch_rows = 5000

for i in range(0, 48000, batch_rows):
    print(f"Iteration{i}")

    t1 = time.time()

    df_tr = pd.read_csv(
        r"/home/vkalekis/Documents/bigdata/dataframes/Test_Preprocessed.csv",
        encoding="utf-8",
        nrows=batch_rows,
        skiprows=i,
    ).fillna(" ")

    X_test = df_tr.iloc[:, -1].values
    X_test = vectorizer.transform(X_test).astype("int8").toarray()

    knn = KNeighborsClassifier(n_jobs=6, n_neighbors=3, metric="jaccard")

    knn.fit(X_train, y_train)
    predictions.extend(knn.predict(X_test))

    t2 = time.time()


df_test = pd.read_csv(
    r"/home/vkalekis/Documents/bigdata/dataframes/test_alex_concat.csv",
    encoding="utf-8",
)

df_pr = pd.DataFrame(
    list(zip(df_test["Id"], predictions)),
    columns=["id", "predicted"],
)

df_pr.to_csv(r"/home/vkalekis/Documents/bigdata/pred.csv", index=True)
