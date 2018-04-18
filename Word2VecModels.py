from gensim.models import KeyedVectors, Word2Vec
from nltk import word_tokenize


def get_glove_embeddings():
    # This is for importing the GloVe data into word2vec format.
    # glove2word2vec('Data\glove.twitter.27B.25d.txt', 'Data\word2vec_twitter.txt')
    w2v_model = KeyedVectors.load_word2vec_format('Data\word2vec_twitter_50.txt')
    #w2v_model = KeyedVectors.load_word2vec_format('D:/Class/ToxicCommentsClassifier/Data/word2vec_twitter.txt')
    embeddings_index = dict()
    for word in w2v_model.wv.vocab:
        embeddings_index[word] = w2v_model.word_vec(word)
    return embeddings_index


def get_trained_embeddings(x_train):
    tokenized_x_train = [word_tokenize(sent) for sent in x_train]
    w2v_model = Word2Vec(tokenized_x_train, size=100, min_count=10, workers=4)
    embeddings_index = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))
    return embeddings_index
