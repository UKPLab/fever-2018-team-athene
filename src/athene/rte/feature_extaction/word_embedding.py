
from nltk import word_tokenize
import numpy as np
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

np.random.seed(42)

def generating_word2vec_batch(model,heads,bodies,vector_length):

    heads_tokens = [[word for word in word_tokenize(head) if word not in stop_words] for head in heads]
    bodies_tokens = [[word for word in word_tokenize(body) if word not in stop_words] for body in bodies]
    # max_head_tokens = max([len(tokens) for tokens in heads_tokens])
    # max_body_tokens = max([len(tokens) for tokens in bodies_tokens])
    head_length = 30
    body_length = 200
    X_heads = np.zeros(shape=(head_length,len(heads),vector_length))
    X_bodies = np.zeros(shape=(body_length,len(bodies),vector_length))
    vocab = model.vocab.keys()
    for idx1,tokens in enumerate(heads_tokens):
        for idx0,token in enumerate(tokens):
            if token in vocab:
                X_heads[idx0,idx1,:] = model.get_vector(token)
            else:
                X_heads[idx0,idx1,:] = np.random.normal(0,0.11,vector_length)
            if idx0 >= head_length-1:
                break

    for idx1,tokens in enumerate(bodies_tokens):
        for idx0,token in enumerate(tokens):
            if token in vocab:
                X_bodies[idx0,idx1,:] = model.get_vector(token)
            else:
                X_bodies[idx0,idx1,:] = np.random.normal(0,0.11,vector_length)
            if idx0 >= body_length-1:
                break


    return X_heads,X_bodies
