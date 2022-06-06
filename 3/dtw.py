"""
    Our implementation for finding the DTW distance between sequences.
    We provide 2 versions of this algorithm:
    1. The no-tagged version which returns the DTW distance, the cost matrix and the traceback matrix.
        Afterwards we can trace back in the warping path to find the best matching between two elements of those sequences.
    2. The banchmarks version which only returns the DTW distance and is applied in the test dataframe.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import time


def initializeMatrices(seq1, seq2, seq1_length, seq2_length):
    """
    Initializes distances, cost and traceback matrix and calculates the elements of the distance matrix.
    """
    distances = np.zeros((seq1_length, seq2_length))

    for i in range(seq1_length):
        for j in range(seq2_length):
            distances[i, j] = np.abs(seq1[i] - seq2[j])
            # distances[i,j] = np.power((seq1[i]-seq2[j]), 2)
            # distances[i,j] = np.sqrt(np.abs(seq1[i]-seq2[j]))
    cost_matrix = np.zeros((seq1_length + 1, seq2_length + 1))
    cost_matrix[0, :] = np.inf
    cost_matrix[:, 0] = np.inf
    cost_matrix[0, 0] = 0

    traceback = np.zeros((seq1_length, seq2_length))

    return distances, cost_matrix, traceback


def initializeMatrices_benchmark(seq1, seq2, seq1_length, seq2_length):
    """
    -- Benchmarks version --
    Initializes distances and cost matrix and calculates the elements of the distance matrix.
    """
    distances = np.zeros((seq1_length, seq2_length))

    for i in range(seq1_length):
        for j in range(seq2_length):
            distances[i, j] = np.abs(seq1[i] - seq2[j])

    cost_matrix = np.zeros((seq1_length + 1, seq2_length + 1))
    cost_matrix[0, :] = np.inf
    cost_matrix[:, 0] = np.inf
    cost_matrix[0, 0] = 0

    return distances, cost_matrix


def dtw(seq1, seq2, seq1_length, seq2_length):
    """
    Implementation of a DTW algorithm given 2 sequences seq1 and seq2.
    Returns DTW distance, cost and traceback matrix.
    """
    distances, cost_matrix, traceback = initializeMatrices(
        seq1, seq2, seq1_length, seq2_length
    )

    for i in range(1, seq1_length + 1):
        for j in range(1, seq2_length + 1):
            candidates = np.array(
                [
                    cost_matrix[i - 1, j - 1],  # Index 0 - Match
                    cost_matrix[i - 1, j],  # Index 1 - Insertion
                    cost_matrix[i, j - 1],
                ]
            )  # Index 2 - Deletion

            cost_matrix[i, j] = distances[i - 1, j - 1] + np.min(candidates)
            traceback[i - 1, j - 1] = np.argmin(candidates)

    return cost_matrix[seq1_length, seq2_length], cost_matrix, traceback


def dtw_benchmark(seq1, seq2, seq1_length, seq2_length):
    """
     \-- Benchmarks version --
    Implementation of a DTW algorithm given 2 sequences seq1 and seq2.
    Returns only DTW distance.
    """
    distances, cost_matrix = initializeMatrices_benchmark(
        seq1, seq2, seq1_length, seq2_length
    )

    for i in range(1, seq1_length + 1):
        for j in range(1, seq2_length + 1):
            candidates = np.array(
                [
                    cost_matrix[i - 1, j - 1],  # Index 0 - Match
                    cost_matrix[i - 1, j],  # Index 1 - Insertion
                    cost_matrix[i, j - 1],
                ]
            )  # Index 2 - Deletion

            cost_matrix[i, j] = distances[i - 1, j - 1] + np.min(candidates)

    return cost_matrix[seq1_length, seq2_length]


def traceback_path(traceback, seq1_length, seq2_length):
    """
    Implements traceback logic. Walks backwards in the warping path to find the best matching
    between the elements of seq1 and seq2.
    """
    current_pos = (seq1_length - 1, seq2_length - 1)
    steps = [current_pos]

    while current_pos != (0, 0):
        direction = traceback[current_pos[0], current_pos[1]]

        if int(direction) == 0:
            # Index 0 - Match
            next_pos = (current_pos[0] - 1, current_pos[1] - 1)
        elif int(direction) == 1:
            # Index 1 - Insertion
            next_pos = (current_pos[0] - 1, current_pos[1])
        elif int(direction) == 2:
            # Index 2 - Deletion
            next_pos = (current_pos[0], current_pos[1] - 1)

        steps.append(next_pos)
        current_pos = next_pos

    return steps


def example():
    """
    Example test using 2 simple sequences.
    """

    # seq1 = np.array([0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0])
    # seq2 = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -0.5, 0, 0])

    seq1 = np.array([1, 1, 2, 3, 2, 0])
    seq2 = np.array([0, 1, 1, 2, 3, 2, 1])

    seq1_length = len(seq1)
    seq2_length = len(seq2)

    dtw_dist, cost_matrix, traceback = dtw(seq1, seq2, seq1_length, seq2_length)
    path = traceback_path(traceback, seq1_length, seq2_length)

    print(dtw_dist)

    x_steps, y_steps = zip(*path)
    plt.plot(x_steps, y_steps)
    plt.show()


def run_on_csv():
    """
    Runs benchmarks on dtw_test.csv
    """

    t1 = time.time()

    df = pd.read_csv("./dtw_test.csv", encoding="utf-8").head(10)

    df["series_a"] = df["series_a"].apply(lambda row: np.array(ast.literal_eval(row)))
    df["series_b"] = df["series_b"].apply(lambda row: np.array(ast.literal_eval(row)))

    df["distance"] = df.apply(
        lambda row: dtw_benchmark(
            row["series_a"], row["series_b"], len(row["series_a"]), len(row["series_b"])
        ),
        axis=1,
    )

    output = df.loc[:, ["id", "distance"]]
    output.to_csv("./output.csv", index=False)

    t2 = time.time() - t1

    print(t2)
    with open("/home/vkalekis/projects/bigdara/src3/test.txt", "w") as f:
        f.write(t2)


if __name__ == "__main__":
    example()
    # run_on_csv()
