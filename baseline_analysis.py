import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from ArrayCaster import ArrayCaster
from PunctuationExtractor import PunctuationExtractor
from SentenceEmbeddingVectorizer import SentenceEmbeddingVectorizer
from UpperCaseExtractor import CapitalExtractor
from Word2VecModels import get_glove_embeddings, get_trained_embeddings

path = 'Data\\train.csv'
# path = 'D:/Class/ToxicCommentsClassifier/Data/train.csv'
train = pd.read_csv(path)
train.drop('id', axis=1, inplace=True)
x_train = train['comment_text']

levels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[levels]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42)

# x_train = x_train[:300]
# y_train = y_train[:300]
# x_test = x_test[:300]
# y_test = y_test[:300]

glove_embeddings_index = get_glove_embeddings()
w2v_trained_embeddings = get_trained_embeddings(x_train)
W2V_MODEL = 'GLOVE'

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=20,
                                   lowercase=False)

classifiers = {'SVC': LinearSVC(random_state=42), 'Logistic Regression': LogisticRegression(random_state=42)}

for name, classifier in classifiers.items():
    ppl = Pipeline([
        ('feats', FeatureUnion([
            # ('ngram', CountVectorizer(ngram_range=(1, 2), analyzer='char', min_df = 30)),
            ('Caps', Pipeline([
                ('Cap', CapitalExtractor()),
                ('caster', ArrayCaster())
            ])),
            ("word2vec vectorizer",
             SentenceEmbeddingVectorizer(glove_embeddings_index) if W2V_MODEL == 'GLOVE' else w2v_trained_embeddings),
            ('tfidf', tfidf_vectorizer),
            ('Punc', Pipeline([
                ('Pun', PunctuationExtractor()),
                ('cast', ArrayCaster())
            ]))
            # ('Punc', PunctuationExtractor())
        ])),
        ('clf', OneVsRestClassifier(classifier))
    ])
    model = ppl.fit(x_train, y_train)
    pred = model.predict(x_test)

    print('Done')
    print('Accuracy for the {1} is {0}'.format(accuracy_score(y_test, pred), name))
    print('Classification report for the {1}: {0}'.format(classification_report(y_test, pred), name))
    print('Hamming Loss for the {1} model is {0}'.format(hamming_loss(y_test, pred), name))
