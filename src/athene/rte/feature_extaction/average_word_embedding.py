from gensim.models import KeyedVectors
import numpy as np
import nltk


# model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)
#
# vecab = model.vocab.keys()
# print(len(vecab))
# vector = model.get_vector('This')
# print(type(vector))
# print(vector.shape)


def load_model(filepath):
    return KeyedVectors.load_word2vec_format(filepath, binary=True)


def average_word_embedding(model, vocab, tokens):
    vectors = []
    count = 0

    for word in tokens:
        if word in vocab:
            count += 1
            vectors.append(model.get_vector(word))
        else:
            continue
    numpy_vectors = np.reshape(np.array(vectors), newshape=(count, 300))

    return np.sum(numpy_vectors, axis=0) / count if count != 0 else np.zeros((1, 300))


def get_word_embedding_features(texts):
    model = load_model('data/GoogleNews-vectors-negative300.bin')
    vocab = model.vocab.keys()
    embedding_features = np.zeros(shape=(len(texts), 300))

    for idx, text in enumerate(texts):
        tokens = nltk.word_tokenize(text, language='english')
        embedding_features[idx] = average_word_embedding(model, vocab, tokens)

    return embedding_features


texts = ['I love somebody!']
