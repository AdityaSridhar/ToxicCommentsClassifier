import pandas as pd
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
#import w2v
import string
from nltk import word_tokenize

#punctuations = string.punctuation
punctuations = ['!', '?', '.']
embeddings_index = dict()

class PunctuationExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def getPuncCount(self, sen):
        count = 0
        #print(sen)
        for char in sen:
            if char in punctuations:
                count +=  1
        return count

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.getPuncCount)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        print(self)
        return self

class CapitalExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def getCapCount(self, sen):
        count = 0
        #print('Caps', sen)
        for char in sen:
            if char.isupper():
                count += 1
        #print(count)
        return count

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.getCapCount)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        #print(self)
        return self


class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    print(data.shape)
    print(np.transpose(np.matrix(data)).shape)
    return np.transpose(np.matrix(data))


# https://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        #print(X)
        X = [word_tokenize(x) for x in X]
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])




'''
def getPuncCount(sen):
    count = 0
    for char in sen:
        if char in punctuations:
            count +=  1
    return count

def getCapCount(sen):
    count = 0
    for char in sen:
        if char.isupper():
            count += 1
    return count
'''
#path = '\Data\train.csv'
punc = []
caps = []
path = 'D:/Class/ToxicCommentsClassifier/Data/train.csv'
train = pd.read_csv(path)
train.drop('id', axis=1, inplace=True)
x_train = train['comment_text']
#for sen in x_train:
#    punc.append(getPuncCount(sen))
#    caps.append(getCapCount(sen))
levels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

y_train = train[levels]

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2), min_df = 15, lowercase = False)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42)
tokenized_x_train = [word_tokenize(sent) for sent in x_train]
tokenized_x_test = [word_tokenize(sent) for sent in x_test]



MODEL = 'GloVe'

if MODEL == 'GLOVE':
    # This is for importing the GloVe data into word2vec format.
    # glove2word2vec('Data\glove.twitter.27B.25d.txt', 'Data\word2vec_twitter.txt')
    #w2v_model = KeyedVectors.load_word2vec_format('Data\word2vec_twitter_50.txt')
    w2v_model = KeyedVectors.load_word2vec_format('D:/Class/ToxicCommentsClassifier/Data/word2vec_twitter_50.txt')
    for word in w2v_model.wv.vocab:
        embeddings_index[word] = w2v_model.word_vec(word)
else:
    w2v_model = Word2Vec(tokenized_x_train, size=100, min_count=10, workers=4)
    embeddings_index = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))


#embeddings_index = dict()
#for word in w2v_model.wv.vocab:
#    embeddings_index[word] = w2v_model.word_vec(word)

ppl = Pipeline([
    ('feats', FeatureUnion ([
        #('ngram', CountVectorizer(ngram_range=(1, 2), analyzer='char', min_df = 30)),
        ('Caps', Pipeline([
            ('Cap', CapitalExtractor()),
            ('caster', ArrayCaster())
            ])),
        ("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings_index)),
        ('tfidf', vectorizer),
        ('Punc', Pipeline ([
            ('Pun', PunctuationExtractor()),
            ('cast', ArrayCaster())
            ]))
        #('Punc', PunctuationExtractor())
        ])),
    ('clf', OneVsRestClassifier(LinearSVC(random_state=42)))
    ])



#x_train = x_train[:300]
#y_train = y_train[:300]
#x_test = x_test[:300]
#y_test = y_test[:300]
model = ppl.fit(x_train, y_train)
pred = model.predict(x_test)

print ('Done')
print(accuracy_score(y_test, pred))




'''
#y_train['punc'] = punc
#y_train['caps'] = caps
#print(x_train[1])
#t_x = []
#for i,x in enumerate(x_train):
#    t_x.append((x, punc[i], caps[i]))

#x_train = t_x
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2), min_df = 20, lowercase = False)
x_train = vectorizer.fit_transform(x_train)#, punc, caps)
#punc = vectorizer.fit_transform(punc)
#caps = vectorizer.fit_transform(caps)

t_x = FeatureUnion([('vectorized', x_train), ('punc', punc), ('caps', caps)])

x_train, x_test, y_train, y_test = train_test_split(t_x, y_train, random_state=42)

x_train = x_train.fit_transform(x_train)
print('ASD', x_train[0])
classifier = OneVsRestClassifier(MultinomialNB())
classifier.fit(x_train, y_train)
#
# test = pd.read_csv('test.csv')
# test.drop('id', axis=1, inplace=True)mul
# x_test = test['comment_text']
x_test = vectorizer.transform(x_test)
# y_test = test[levels]

predictions = classifier.predict(x_test)
print(accuracy_score(y_test, predictions))
# print(precision_score(y_test, predictions, average=))
# print(recall_score(y_test, predictions))

# clf = MLPClassifier(hidden_layer_sizes=(5, 10), random_state=42)
# clf.fit(x_train, y_train)
    
clf = OneVsRestClassifier(LinearSVC(random_state=42))
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print(accuracy_score(y_test, pred))
'''

