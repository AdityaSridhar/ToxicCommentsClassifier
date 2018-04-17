from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ArrayCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        print(data.shape)
        print(np.transpose(np.matrix(data)).shape)
        return np.transpose(np.matrix(data))
