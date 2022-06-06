from datasketch import MinHash, MinHashLSH
import time


def preprocess(text):
    """
    Split sentences to words.
    """
    tokens = text.split()
    return tokens


def create_minhash(sentence, perms):
    """
    Create MinHash object for sentence and permutations provided.
    """
    tokens = preprocess(sentence)
    m = MinHash(num_perm=perms)
    for token in tokens:
        m.update(token.encode("utf8"))
    return m


def get_forest(data, perms, threshold):
    """
    Create a forest of MinHashed value for data by providing permutation and threshold values.
    """
    start_time = time.time()

    minhashes = []

    for text in data:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for token in tokens:
            m.update(token.encode("utf8"))

        minhashes.append(m)

    forest = MinHashLSH(threshold=threshold, num_perm=perms)

    for i, minhash in enumerate(minhashes):
        forest.insert(i, minhash)

    return forest, time.time() - start_time


def queryForest(X_test, forest, permutations):
    """
    MinHash Test dataset and query forest. Returns the most similar results from forest, based on threshold value.
    """
    queries = X_test
    results = []

    start_time = time.time()

    for query in queries:
        query_minhash = create_minhash(query, permutations)
        returned_query = forest.query(query_minhash)
        results.append(returned_query)

    return results, time.time() - start_time
