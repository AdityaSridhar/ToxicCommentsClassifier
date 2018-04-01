import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

train = pd.read_csv('\Data\train.csv')
train.drop('id', axis=1, inplace=True)
x_train = train['comment_text']
levels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[levels]
print(x_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
x_train = vectorizer.fit_transform(x_train)

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


