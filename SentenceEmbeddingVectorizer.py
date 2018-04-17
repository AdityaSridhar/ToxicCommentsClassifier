from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# Adapted from https://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class SentenceEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = [word_tokenize(x) for x in X]
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
