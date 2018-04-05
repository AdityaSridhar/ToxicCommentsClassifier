from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

# glove2word2vec('Data\glove.twitter.27B.25d.txt', 'Data\word2vec_twitter.txt')
from sklearn.svm import LinearSVC

w2v_model = KeyedVectors.load_word2vec_format('Data\word2vec_twitter.txt')
print(w2v_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))


# https://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


embeddings_index = dict()
for word in w2v_model.wv.vocab:
    embeddings_index[word] = w2v_model.word_vec(word)

etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings_index)),
    ("svc", OneVsRestClassifier(LinearSVC(random_state=42)))])

train = pd.read_csv('Data\\train.csv')
train.drop('id', axis=1, inplace=True)
x_train = train['comment_text']
levels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

y_train = train[levels]
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42)

model = etree_w2v.fit(x_train, y_train)
pred = model.predict(x_test)
print('Done')
print(accuracy_score(y_test, pred))
