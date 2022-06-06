
* Requirement 1

    * /graphs : figures/graphs used in the report 
    * Source files :
        * _jaccard_tests.ipynb_ : Jupyter notebook file for the tests using different Jaccard metrics and different representations for the documents.
        * _knn_ontest.py_ : Python file for running predictions using a KNN classifier on the unknown Kaggle test set.
        * _knn_preprocess.py_ : Python file for the preprocess pipeline on the documents using the nltk library.
        * _knn.py_ : Python file for splitting the train dataset into train and test sub-datasets and running the KNN classifer to make predictions on the test sub-dataset.
        * _knnkfold.py_ : Python file for running predictions on test sub-datasets by k-folding the original train dataset.
        * _ml.ipynb_ : Jupyter notebook file for the development of a neural network for text classification.
        * _representation_tests.ipynb_ : Jupyter notebook file for generating the different representations of the two example sentences in the report.
        * _WordCloud.py_ : Python file for the WordCloud visualizations for the 4 categories of documents.
* Requirement 2
    
    * graphs : figures/graphs used in the report 
    * Source files :
        * _graphs.ipynb_ : Jupyter notebook file for generating the graphs used in the report.
        * _lsh_global.py_ : Python file for implementing a KNN-LSH classifier with Global Vectorizer. The train set is splitted to train and test sub-datasets and the KNN-LSH algorithm is evaluated based on the accuracy and time on the test set using different permutations and thresholds for the LSH algorithm.
        * _lsh_local.py_ : Python file for implementing a KNN-LSH classifier with Local Vectorizer. The train set is splitted to train and test sub-datasets and the KNN-LSH algorithm is evaluated based on the accuracy and time on the test set using different permutations and thresholds for the LSH algorithm.
        * _lsh_modules.py_ : Python file which implements the building and quering procedures on a LSH forest.
        * _lsh_ontestset.py_ : Python file which used the KNN-LSH algorithm with a given permutation number and threshold on the unknown test set.
        * _lsh_preprocessing.py_ : Python file for the preprocessing pipeline used in KNN+LSH (Same as the one in Requirement 1).

* Requirement 3
    
    * /graphs : figures/graphs used in the report 
    * Source files :
        * _dtw.py_ : Python file for our implementation of the DTW algorithm. Contains 2 different versions:
            * The normal/untagged version which calculates the DTW distance and the warping path of two sequences.
            * The benchmarks version which is applied on the dataframe and only calculates the DTW distance between two sequences.
        * _dtw_demo.ipynb_ : Jupyter notebook file for demonstrating the use of DTW on 3 small sequences and 2 sequences on the Kaggle test set. For the 3 small sequences, different graphs are generated which show the traditional and DTW matching and a heatmap of the cost matrix with the warping path.


