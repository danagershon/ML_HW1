import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy

class kNN(BaseEstimator, ClassifierMixin):
  def __init__(self, n_neighbors:int = 3):
    self.n_neighbors = n_neighbors

  def fit(self, X, y):
    self.X = np.copy(X)
    self.y = np.copy(y)

  def predict(self, X):
    distances = scipy.spatial.distance.cdist(X, self.X)
    arg_neighbors = np.argpartition(distances, self.n_neighbors, axis=1)
    label_of_neighbors = self.y[arg_neighbors[:,:self.n_neighbors]]
    predictions = np.sign(np.sum(label_of_neighbors, axis=1))
    return predictions