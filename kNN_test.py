import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import unittest

import kNN

class KNNNotebookTest(unittest.TestCase):

    def setUp(self) -> None:
        print("\nKNNNotebookTest:")
        # load data from csv
        self.dataset = pd.read_csv('virus_data.csv')
        # split to train and test sets
        self.df_train, self.df_test = train_test_split(self.dataset, train_size=0.8, random_state=74+40)

        # keep only chosen pair PCR features and spread label
        self.df_train_features_pair = self.df_train[["PCR_04", "PCR_09"]]
        self.df_train_spread_labels = self.df_train["spread"]

        self.df_test_features_pair = self.df_test[["PCR_04", "PCR_09"]]
        self.df_test_spread_labels = self.df_test["spread"]

        #for Q11-12 - normalize the features pair using MinMax scaling between [-1,1]
        self.df_train_features_pair_normalized = self.df_train_features_pair.copy()
        self.df_test_features_pair_normalized = self.df_test_features_pair.copy()

        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(self.df_train_features_pair_normalized) # ensure the min and max values are from the train set

        # normalize the train set
        self.df_train_features_pair_normalized[self.df_train_features_pair_normalized.columns] = scaler.transform(self.df_train_features_pair_normalized[self.df_train_features_pair_normalized.columns])
        # normalize the test set
        self.df_test_features_pair_normalized[self.df_test_features_pair_normalized.columns] = scaler.transform(self.df_test_features_pair_normalized[self.df_test_features_pair_normalized.columns])

    def print_accuracy(self, knn_model, df, labels, type):
        acc = knn_model.score(df, labels)
        print(f"{knn_model.n_neighbors}-NN {type} accuracy is: {acc*100:.2f}%")

    def fit_predict(self, knn_model, train_set, train_label, test_set, test_label, normalized):
        knn_model.fit(train_set, train_label)
        self.print_accuracy(knn_model, train_set, train_label, ("not" if not normalized else '') + " normalized train")
        self.print_accuracy(knn_model, test_set, test_label, ("not" if not normalized else '') + " normalized test")

    def test_q10(self):
        #Q10 - 1-NN without normaliztion
        self.fit_predict(kNN.kNN(1),
                         self.df_train_features_pair, self.df_train_spread_labels, 
                         self.df_test_features_pair, self.df_test_spread_labels,
                         normalized=False)

    def test_q11(self):
        #Q11 - train 1-NN model with the normalized dataset
        self.fit_predict(kNN.kNN(1),
                         self.df_train_features_pair_normalized, self.df_train_spread_labels,
                         self.df_test_features_pair_normalized, self.df_test_spread_labels,
                         normalized=True)

    def test_q12(self):
        #Q12 - train 5-NN model with the normalized dataset
        self.fit_predict(kNN.kNN(5),
                         self.df_train_features_pair_normalized, self.df_train_spread_labels,
                         self.df_test_features_pair_normalized, self.df_test_spread_labels,
                         normalized=True)

class TestKNNBase():
        
    def fit_predict(self, knn):
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.shape, "Prediction shape mismatch")
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"{knn.n_neighbors}-NN test accuracy: {accuracy*100:.2f}%")
    
    def test_neighbors_parameter(self):
        max_n_neighbors = 5
        for n_neighbors in range(1, max_n_neighbors + 1):
            knn_model = kNN.kNN(n_neighbors)
            self.fit_predict(knn_model)

class SimpleDatasetKNNTest(TestKNNBase, unittest.TestCase):

    def setUp(self):
        print("\nSimpleDatasetKNNTest:")
        # Create a simple dataset
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=8, n_redundant=2, n_classes=2, 
            flip_y=0.1, class_sep=0.5, random_state=42
        )
        y = 2 * y - 1  # Convert labels to -1 and 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class ToyDatasetKNNTest(TestKNNBase, unittest.TestCase):

    def setUp(self):
        print("\nToyDatasetKNNTest:")
        # Load the Breast Cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Convert labels to -1 and 1
        y = 2 * y - 1
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)


if __name__ == '__main__':
    unittest.main()