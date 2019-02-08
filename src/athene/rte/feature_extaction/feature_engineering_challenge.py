import os
import os.path as path
import pickle
import string
from datetime import datetime
from time import time

import nltk
import numpy as np
import regex as re
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from fnc.refs.utils.generate_test_splits import kfold_split
from sklearn.metrics.pairwise import cosine_distances

from fnc.settings import myConstants
from fnc.utils.loadEmbeddings import LoadEmbeddings

_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def clear_unwanted_chars(mystring):
    return str(mystring.encode('latin', errors='ignore').decode('latin'))


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file, bodyId, feature, headId=""):
    if not os.path.isfile(feature_file):
        if 'stanford' in feature:
            feats = feat_fn(headlines, bodies, bodyId, headId)
        else:
            feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


def gen_BOW_feats(feat_fn, headlines, bodies, headlines_test, bodies_test, features_dir, feature,
                  fold):
    feature_file = "%s/%s.%s.npy" % (features_dir, feature, fold)
    if not os.path.isfile(feature_file):
        print(str(datetime.now()) + ": Generating features for: " + feature + ", fold/holdout: " + str(fold))

        X_train, X_test = feat_fn(headlines, bodies, headlines_test, bodies_test)

        if (str(fold) != 'holdout'):
            np.save("%s/%s.%s.npy" % (features_dir, feature, fold), X_train)
            np.save("%s/%s.%s.test.npy" % (features_dir, feature, fold), X_test)
        else:
            np.save("%s/%s.%s.npy" % (features_dir, feature, 'holdout'), X_train)
            np.save("%s/%s.%s.test.npy" % (features_dir, feature, 'holdout'), X_test)


def load_embeddings(headlines, bodies):
    # embedding parameters:
    embedding_size = 300
    vocab_size = 3000000
    embeddPath = "%s/data/embeddings/google_news/GoogleNews-vectors-negative300.bin.gz" % myConstants.BASE_DIR
    embeddData = path.normpath("%s/data/" % (path.dirname(path.abspath(embeddPath))))
    binary_val = True
    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size,
                                embedding_size=embedding_size, binary_val=binary_val)
    #     print('Loaded embeddings: Vocab-Size: ' + str(vocab_size) + ' \n Embedding size: ' + str(embedding_size))
    return embedding_size, embeddings


def get_head_body_tuples_test():
    d = myConstants.testdataset

    h = []
    b = []
    for stance in d.stances:
        h.append(stance['Headline'])
        b.append(d.articles[int(stance['Body ID'])])

    return h, b


def get_head_body_tuples(include_holdout=False):
    # file paths
    '''
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    splits_dir = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    dataset = DataSet(data_path)
    '''
    data_path = myConstants.data_path
    splits_dir = myConstants.splits_dir
    dataset = myConstants.d

    def get_stances(dataset, folds, holdout):
        # Creates the list with a dict {'headline': ..., 'body': ..., 'stance': ...} for each
        # stance in the data set (except for holdout)
        stances = []
        for stance in dataset.stances:
            if stance['Body ID'] in holdout and include_holdout == True:
                stances.append(stance)
            for fold in folds:
                if stance['Body ID'] in fold:
                    stances.append(stance)

        return stances

    # create new vocabulary
    folds, holdout = kfold_split(dataset, n_folds=10,
                                 base_dir=splits_dir)  # [[133,1334,65645,], [32323,...]] => body ids for each fold
    stances = get_stances(dataset, folds, holdout)

    print("Stances length: " + str(len(stances)))

    h = []
    b = []
    # create the final lists with all the headlines and bodies of the set except for holdout
    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    return h, b


def create_word_ngram_vocabulary(ngram_range=(1, 1), max_features=100, lemmatize=False, term_freq=False, norm='l1',
                                 use_idf=False, include_holdout=False):
    """
    Creates, returns and saves a vocabulary for (Count-)Vectorizer over all training and test data (holdout excluded) to create BoW
    methods. The method simplifies using the pipeline and later tests with feature creation for a single headline and body.
    This method will cause bleeding, since it also includes the test set.

    :param filename: a filename for the vocabulary
    :param ngram_range: the ngram range for the Vectorizer. Default is (1, 1) => unigrams
    :param max_features: the length of the vocabulary
    :return: the vocabulary
    """
    # file paths
    '''
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    splits_dir = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    dataset = DataSet(data_path)
    '''
    features_dir = myConstants.features_dir

    print("Calling create_word_ngram_vocabulary with ngram_range=("
          + str(ngram_range[0]) + ", " + str(ngram_range[1]) + "), max_features="
          + str(max_features) + ", lemmatize=" + str(lemmatize) + ", term_freq=" + str(term_freq))

    def get_all_stopwords():
        stop_words_nltk = set(stopwords.words('english'))  # use set for faster "not in" check
        stop_words_sklearn = feature_extraction.text.ENGLISH_STOP_WORDS
        all_stop_words = stop_words_sklearn.union(stop_words_nltk)
        return all_stop_words

    def get_tokenized_lemmas_without_stopwords(s):
        all_stop_words = get_all_stopwords()
        return [normalize_word(t) for t in nltk.word_tokenize(s)
                if t not in string.punctuation and t.lower() not in all_stop_words]

    def train_vocabulary(head_and_body):
        # trains a CountVectorizer on all of the data except for holdout data
        if lemmatize == False:
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features,
                                             use_idf=use_idf, norm=norm)
        else:
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features,
                                         tokenizer=get_tokenized_lemmas_without_stopwords)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                                             tokenizer=get_tokenized_lemmas_without_stopwords, use_idf=use_idf,
                                             norm=norm)
        vectorizer.fit_transform(head_and_body)
        vocab = vectorizer.vocabulary_
        return vocab

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    # create filename for vocab
    vocab_file = "word_(" + str(ngram_range[0]) + "_" + str(ngram_range[1]) + ")-gram_" + str(max_features)
    if lemmatize == True:
        vocab_file += "_lemmatized"
    if term_freq == True:
        vocab_file += "_tf"
    if use_idf == True:
        vocab_file += "_idf"
    if include_holdout == True:
        vocab_file += "_holdout"
    vocab_file += "_" + norm + ".pickle"

    # if vocab already exists, just load and return it
    if (os.path.exists(features_dir + "/" + vocab_file)):
        with open(features_dir + "/" + vocab_file, 'rb') as handle:
            vocab = pickle.load(handle)
            print("Existing vocabulary found and load.")
            return vocab

    h, b = get_head_body_tuples(include_holdout=include_holdout)
    head_and_body = combine_head_and_body(h, b)  # combine head and body
    vocab = train_vocabulary(head_and_body)  # get vocabulary (features)

    # save the vocabulary as file
    with open(features_dir + "/" + vocab_file, 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("vocab length: " + str(len(vocab)))
    return vocab


def NMF_fit_all_incl_holdout_and_test(headlines, bodies):
    # http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    from sklearn.externals import joblib

    print("WARNING: IF SIZE OF HEAD AND BODY DO NOT MATCH, "
          "RUN THIS FEATURE EXTRACTION METHOD SEPERATELY (WITHOUT ANY OTHER FE METHODS) TO CREATE THE FEATURES ONCE!")

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_incl_holdout_and_test: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_incl_holdout_and_test: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)

    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_incl_holdout_and_test: fit and transform body")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)

        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_incl_holdout_and_test: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_incl_holdout_and_test: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(nfm_head_matrix)):
            X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1))  # 1d array is deprecated
            X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()
    h.extend(h_test)
    b.extend(b_test)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X


def latent_dirichlet_allocation_incl_holdout_and_test(headlines, bodies):
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
        # more important topic words a body contains of a certain topic, the higher its value for this topic
        lda_body = LatentDirichletAllocation(n_topics=100, learning_method='online', random_state=0, n_jobs=3)

        print("latent_dirichlet_allocation_incl_holdout_and_test: fit and transform body")
        t0 = time()
        lda_body_matrix = lda_body.fit_transform(X_train_body)
        print("done in %0.3fs." % (time() - t0))

        print("latent_dirichlet_allocation_incl_holdout_and_test: transform head")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        lda_head_matrix = lda_body.transform(X_train_head)

        # print_top_words(lda_body, vectorizer_body.get_feature_names(), 100)

        print('latent_dirichlet_allocation_incl_holdout_and_test: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(lda_head_matrix)):
            X_head_vector = np.array(lda_head_matrix[i]).reshape((1, -1))  # 1d array is deprecated
            X_body_vector = np.array(lda_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X

    h, b = get_head_body_tuples(include_holdout=True)

    h_test, b_test = get_head_body_tuples_test()

    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies: " + str(len(b)))
    h.extend(h_test)
    b.extend(b_test)
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads after ext: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies after ext: " + str(len(b)))

    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=False,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h, b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)
    return X


def latent_semantic_indexing_gensim_holdout_and_test(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    from gensim import corpora, models

    def combine_and_tokenize_head_and_body(headlines, bodies, file_path=None):
        all_text = []
        all_text.extend(headlines)
        all_text.extend(bodies)
        if file_path != None and (os.path.exists(file_path)):
            with open(file_path, 'rb') as handle:
                return pickle.load(handle)

        print("head+body appended size should be around 100k and 19/8k: " + str(len(bodies)))
        head_and_body_tokens = [nltk.word_tokenize(line) for line in all_text]

        if file_path != None:
            with open(file_path, 'wb') as handle:
                pickle.dump(head_and_body_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return head_and_body_tokens

    def get_features(n_topics):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR

        filename = "lsi_gensim_test_" + str(n_topics) + "topics_and_test"

        h, b = get_head_body_tuples(include_holdout=True)
        h_test, b_test = get_head_body_tuples_test()
        h.extend(h_test)
        b.extend(b_test)
        head_and_body = combine_and_tokenize_head_and_body(h, b,
                                                           file_path=features_dir + "/" + "lsi_gensim_h_b_tokenized_and_test" + ".pkl")

        if (os.path.exists(features_dir + "/" + "lsi_gensim_holdout_and_test" + ".dict")):
            print("dict found and load")
            dictionary = corpora.Dictionary.load(features_dir + "/" + "lsi_gensim_all_and_test" + ".dict")
        else:
            print("create new dict")
            dictionary = corpora.Dictionary(head_and_body)
            dictionary.save(features_dir + "/" + "lsi_gensim_all_and_test" + ".dict")

        if (os.path.exists(features_dir + "/" + filename + ".lsi")):
            print("found lsi model")
            lsi = models.LsiModel.load(features_dir + "/" + filename + ".lsi")
        else:
            print("build corpus and tfidf corpus")
            corpus = [dictionary.doc2bow(text) for text in head_and_body]
            tfidf = models.TfidfModel(corpus)  # https://stackoverflow.com/questions/6287411/lsi-using-gensim-in-python
            corpus_tfidf = tfidf[corpus]

            print("create new lsi model")
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
            lsi.save(features_dir + "/" + filename + ".lsi")

        # get tfidf corpus of head and body
        corpus_train = [dictionary.doc2bow(text) for text in combine_and_tokenize_head_and_body(headlines, bodies)]
        tfidf_train = models.TfidfModel(corpus_train)
        corpus_train_tfidf = tfidf_train[corpus_train]

        corpus_lsi = lsi[corpus_train_tfidf]

        X_head = []
        X_body = []
        i = 0
        for doc in corpus_lsi:
            if i < int(len(corpus_lsi) / 2):
                X_head_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_head_vector_filled[id] = prob
                X_head.append(X_head_vector_filled)
            else:
                X_body_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_body_vector_filled[id] = prob
                X_body.append(X_body_vector_filled)
            i += 1

        X = np.concatenate([X_head, X_body], axis=1)

        return X

    n_topics = 300
    X = get_features(n_topics)

    return X


def NMF_fit_all_concat_300_and_test(headlines, bodies):
    # http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    from sklearn.externals import joblib

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            print("X_all_length (w Holdout round 50k): " + str(len(head_and_body)))
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300_and_test: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300_and_test: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)

    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_concat_300_and_test: fit NMF to all data")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)

        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_concat_300_and_test: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_concat_300_and_test: concat head and body')
        # calculate cosine distance between the body and head
        return np.concatenate([nfm_head_matrix, nfm_body_matrix], axis=1)

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()
    h.extend(h_test)
    b.extend(b_test)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X


def word_ngrams_concat_tf5000_l2_w_holdout(headlines, bodies):
    """
    Simple bag of words feature extraction
    """

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    vocab = create_word_ngram_vocabulary(ngram_range=(1, 1), max_features=5000,
                                         lemmatize=False, use_idf=False, term_freq=True, norm='l2',
                                         include_holdout=True)

    X = get_features(vocab)

    return X


def word_ngrams_concat_tf5000_l2_w_holdout_and_test(headlines, bodies):
    """
    Simple bag of words feature extraction
    """

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()

    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies: " + str(len(b)))
    h.extend(h_test)
    b.extend(b_test)
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads after ext: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies after ext: " + str(len(b)))

    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=True,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h, b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)

    return X


def NMF_fit_all(headlines, bodies):
    # http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    from sklearn.externals import joblib

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)

    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=50, random_state=1, alpha=.1)

            print("NMF_fit_all: fit and transform body")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)

        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(nfm_head_matrix)):
            X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1))  # 1d array is deprecated
            X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X

    h, b = get_head_body_tuples()
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X


def latent_dirichlet_allocation(headlines, bodies):
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
        # more important topic words a body contains of a certain topic, the higher its value for this topic
        lda_body = LatentDirichletAllocation(n_topics=25, learning_method='online', random_state=0, n_jobs=3)

        print("latent_dirichlet_allocation: fit and transform body")
        t0 = time()
        lda_body_matrix = lda_body.fit_transform(X_train_body)
        print("done in %0.3fs." % (time() - t0))

        print("latent_dirichlet_allocation: transform head")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        lda_head_matrix = lda_body.transform(X_train_head)

        # print_top_words(lda_body, vectorizer_body.get_feature_names(), 100)

        print('latent_dirichlet_allocation: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(lda_head_matrix)):
            X_head_vector = np.array(lda_head_matrix[i]).reshape((1, -1))  # 1d array is deprecated
            X_body_vector = np.array(lda_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X

    vocab = create_word_ngram_vocabulary(ngram_range=(1, 1), max_features=5000, lemmatize=False, term_freq=True,
                                         norm='l2')
    X = get_features(vocab)
    return X


def latent_semantic_indexing_gensim_test(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    from gensim import corpora, models

    def combine_and_tokenize_head_and_body(headlines, bodies, file_path=None):
        all_text = []
        all_text.extend(headlines)
        all_text.extend(bodies)
        if file_path != None and (os.path.exists(file_path)):
            with open(file_path, 'rb') as handle:
                return pickle.load(handle)

        print("head+body appended size should be around 100k and 19/8k: " + str(len(bodies)))
        head_and_body_tokens = [nltk.word_tokenize(line) for line in all_text]

        if file_path != None:
            with open(file_path, 'wb') as handle:
                pickle.dump(head_and_body_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return head_and_body_tokens

    def get_features(n_topics):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR

        filename = "lsi_gensim_test_" + str(n_topics) + "topics"

        h, b = get_head_body_tuples(include_holdout=True)
        head_and_body = combine_and_tokenize_head_and_body(h, b,
                                                           file_path=features_dir + "/" + "lsi_gensim_h_b_tokenized" + ".pkl")

        if (os.path.exists(features_dir + "/" + "lsi_gensim_holdout" + ".dict")):
            print("dict found and load")
            dictionary = corpora.Dictionary.load(features_dir + "/" + "lsi_gensim_all" + ".dict")
        else:
            print("create new dict")
            dictionary = corpora.Dictionary(head_and_body)
            dictionary.save(features_dir + "/" + "lsi_gensim_all" + ".dict")

        if (os.path.exists(features_dir + "/" + filename + ".lsi")):
            print("found lsi model")
            lsi = models.LsiModel.load(features_dir + "/" + filename + ".lsi")
        else:
            print("build corpus and tfidf corpus")
            corpus = [dictionary.doc2bow(text) for text in head_and_body]
            tfidf = models.TfidfModel(corpus)  # https://stackoverflow.com/questions/6287411/lsi-using-gensim-in-python
            corpus_tfidf = tfidf[corpus]

            print("create new lsi model")
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
            lsi.save(features_dir + "/" + filename + ".lsi")

        # get tfidf corpus of head and body
        corpus_train = [dictionary.doc2bow(text) for text in combine_and_tokenize_head_and_body(headlines, bodies)]
        tfidf_train = models.TfidfModel(corpus_train)
        corpus_train_tfidf = tfidf_train[corpus_train]

        corpus_lsi = lsi[corpus_train_tfidf]

        X_head = []
        X_body = []
        i = 0
        for doc in corpus_lsi:
            if i < int(len(corpus_lsi) / 2):
                X_head_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_head_vector_filled[id] = prob
                X_head.append(X_head_vector_filled)
            else:
                X_body_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_body_vector_filled[id] = prob
                X_body.append(X_body_vector_filled)
            i += 1

        X = np.concatenate([X_head, X_body], axis=1)

        return X

    n_topics = 300
    X = get_features(n_topics)

    return X


def NMF_fit_all_concat_300(headlines, bodies):
    # http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    from sklearn.externals import joblib

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_concat_300"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            print("X_all_length (w Holout round 50k): " + str(len(head_and_body)))
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_concat_300"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)

    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % myConstants.BASE_DIR
        filename = "NMF_fit_all_concat_300"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_concat_300: fit NMF to all data")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)

        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_concat_300: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_concat_300: concat head and body')
        # calculate cosine distance between the body and head
        return np.concatenate([nfm_head_matrix, nfm_body_matrix], axis=1)

    h, b = get_head_body_tuples(include_holdout=True)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X
