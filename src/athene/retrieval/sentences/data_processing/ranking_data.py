import os
import pickle
import random

import nltk
import numpy as np
from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB

random.seed(100)
np.random.seed(100)


def get_valid_texts(lines, page):
    if not lines[0]:
        return []
    doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                 lines.split("\n")]
    doc_lines = zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines)))
    return doc_lines


def get_whole_evidence(evidence_set, db):
    pos_sents = []
    for evidence in evidence_set:
        page = evidence[2]
        doc_lines = db.get_doc_lines(page)
        doc_lines = get_valid_texts(doc_lines, page)
        for doc_line in doc_lines:
            if doc_line[2] == evidence[3]:
                pos_sents.append(doc_line[0])
    pos_sent = ' '.join(pos_sents)
    return pos_sent


def in_doc_sampling(db_filename, datapath, num_sample=1):
    db = FeverDocDB(db_filename)
    jlr = JSONLineReader()

    X = []
    count = 0
    with open(datapath, "r") as f:
        lines = jlr.process(f)

        for line in tqdm(lines):
            count += 1
            pos_pairs = []
            # count1 += 1
            if line['label'].upper() == "NOT ENOUGH INFO":
                continue
            neg_sents = []
            claim = line['claim']

            pos_set = set()
            for evidence_set in line['evidence']:
                pos_sent = get_whole_evidence(evidence_set, db)
                if pos_sent in pos_set:
                    continue
                pos_set.add(pos_sent)

            p_lines = []
            evidence_set = set([(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])
            page_set = set([evidence[0] for evidence in evidence_set])
            for page in page_set:
                doc_lines = db.get_doc_lines(page)
                p_lines.extend(get_valid_texts(doc_lines, page))
            for doc_line in p_lines:
                if (doc_line[1], doc_line[2]) not in evidence_set:
                    neg_sents.append(doc_line[0])

            num_sampling = num_sample
            if len(neg_sents) < num_sampling:
                num_sampling = len(neg_sents)
                # print(neg_sents)
            if num_sampling == 0:
                continue
            else:
                for pos_sent in pos_set:
                    samples = random.sample(neg_sents, num_sampling)
                    for sample in samples:
                        if not sample:
                            continue
                        X.append((claim, pos_sent, sample))
                        if count % 1000 == 0:
                            print("claim:{} ,evidence :{} sample:{}".format(claim, pos_sent, sample))
    return X


def in_class_sampling(db_filename, datapath, num_sample=1, k=5):
    """

        :param db_filename: path stores wiki-pages database
        :param datapath: path stores fever predicted pages train set
        :param k: number of sentences where to select negative examples
        :param num_sample: number of negative examples to sample
        :return: X: claim and sentence pairs y: if the sentence in evidence set
        """

    db = FeverDocDB(db_filename)
    jlr = JSONLineReader()

    X = []
    count = 0

    count1 = 1
    with open(datapath, "r") as f:
        lines = jlr.process(f)
        # lines = lines[:1000]

        for line in tqdm(lines):
            pos_pairs = []
            count1 += 1
            num_sampling = num_sample
            if line['label'].upper() == "NOT ENOUGH INFO":
                continue
            p_lines = []
            neg_sents = []
            claim = line['claim']

            for evidence_set in line['evidence']:
                pos_sent = get_whole_evidence(evidence_set, db)
                print("claim:{} pos_sent:{}".format(claim, pos_sent))
                pos_pairs.append((claim, pos_sent))

            evidence_set = set([(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])
            sampled_sents_idx = [(id, number) for id, number in line['predicted_sentences']]
            sampled_sents_idx = sampled_sents_idx[0:k + 5]
            sampled_sents_idx = [index for index in sampled_sents_idx if index not in evidence_set]
            pages = set()
            pages.update(evidence[0] for evidence in line['predicted_pages'])
            pages.update(evidence[0] for evidence in evidence_set)
            for page in pages:
                doc_lines = db.get_doc_lines(page)
                p_lines.extend(get_valid_texts(doc_lines, page))
            for doc_line in p_lines:
                if not doc_line[0]:
                    continue
                elif (doc_line[1], doc_line[2]) in sampled_sents_idx:
                    neg_sents.append(doc_line[0])
                # elif (doc_line[1], doc_line[2]) in evidence_set:
                #     if count1%10000==0:
                #         print("page_id:{},sent_num:{}".format(doc_line[1],doc_line[2]))
                #         print("evidence_set:{}".format(evidence_set))
                #     pos_pairs.append((claim,doc_line[0]))

            if len(sampled_sents_idx) < num_sample:
                num_sampling = len(neg_sents)
            if num_sampling == 0:
                count += 1
                continue
            else:
                for pair in pos_pairs:
                    samples = random.sample(neg_sents, num_sampling)
                    for sample in samples:
                        X.append((pair[0], pair[1], sample))
                        if count1 % 10000 == 0:
                            print("claim:{},pos:{},neg:{}".format(claim, pair[1], sample))
        print(count)

    return X


def dev_processing(db_filename, datapath):
    db = FeverDocDB(db_filename)
    jlr = JSONLineReader()

    devs = []
    all_indexes = []

    with open(datapath, "rb") as f:
        lines = jlr.process(f)

        for line in tqdm(lines):
            dev = []
            indexes = []
            pages = set()
            pages.update(page[0] for page in line['predicted_pages'])
            if len(pages) == 0:
                pages.add("Michael_Hutchence")
            claim = line['claim']
            p_lines = []
            for page in pages:
                doc_lines = db.get_doc_lines(page)
                if not doc_lines:
                    continue
                p_lines.extend(get_valid_texts(doc_lines, page))

            for doc_line in p_lines:
                if not doc_line[0]:
                    continue
                dev.append((claim, doc_line[0]))
                indexes.append((doc_line[1], doc_line[2]))
            # print(len(dev))
            if len(dev) == 0:
                dev.append((claim, 'no evidence for this claim'))
                indexes.append(('empty', 0))
            devs.append(dev)
            all_indexes.append(indexes)
    return devs, all_indexes


# def out_class_sampling():
#     pass


def train_data_loader(train_sampled_path, db_filename, data_path, num_samples=1, k=5, sampling='doc'):
    if os.path.exists(train_sampled_path):
        with open(train_sampled_path, 'rb') as f:
            X = pickle.load(f)
    else:
        if sampling == 'similar_sentences':
            X = in_class_sampling(db_filename, data_path, num_samples, k)
            with open(train_sampled_path, 'wb') as f:
                pickle.dump(X, f)
        elif sampling == "same_doc":
            X = in_doc_sampling(db_filename, data_path, num_samples)
            with open(train_sampled_path, 'wb') as f:
                pickle.dump(X, f)
    return X


def dev_data_loader(dev_data_path, db_filename, data_path):
    if os.path.exists(dev_data_path):
        print(dev_data_path)
        with open(dev_data_path, "rb") as f:
            data = pickle.load(f)
            devs, location_indexes = zip(*data)
    else:
        devs, location_indexes = dev_processing(db_filename, data_path)
        data = zip(devs, location_indexes)
        with open(dev_data_path, 'wb') as f:
            pickle.dump(data, f)
    return devs, location_indexes


def sent_processing(sent):
    sent = sent.replace('\n', '')
    sent = sent.replace('-', ' ')
    sent = sent.replace('/', ' ')
    return sent


def corenlp_tokenizer(sent, nlp):
    # print(nlp.word_tokenize(sent))
    return nlp.word_tokenize(sent)


def simple_tokenizer(sent):
    return sent.strip().split(' ')


def nltk_tokenizer(sent):
    # sent = sent_processing(sent)
    return nltk.word_tokenize(sent)


def get_words(claims, sents, h_max_length, s_max_length):
    words = set()
    for claim in claims:
        for idx, word in enumerate(nltk_tokenizer(claim)):
            if idx >= h_max_length:
                continue
            words.add(word.lower())
    for sent in sents:
        for idx, word in enumerate(nltk_tokenizer(sent)):
            if idx >= s_max_length:
                continue
            words.add(word.lower())
    return words


def get_train_words(X, h_max_length, s_max_length):
    claims = set()
    sents = []
    # nlp = StanfordCoreNLP(corenlp_path)
    for claim, pos, neg in X:
        claims.add(claim)
        sents.append(pos)
        sents.append(neg)

    train_words = get_words(claims, sents, h_max_length, s_max_length)
    # nlp.close()
    return train_words


def get_dev_words(devs, h_max_length, s_max_length):
    dev_words = set()
    # nlp = StanfordCoreNLP(corenlp_path)
    for dev in tqdm(devs):
        claims = set()
        sents = []
        for pair in dev:
            claims.add(pair[0])
            sents.append(pair[1])
        dev_tokens = get_words(claims, sents, h_max_length, s_max_length)
        dev_words.update(dev_tokens)
    print("dev_words processing done!")
    # nlp.close()
    return dev_words


def word_2_dict(words):
    word_dict = {}
    for idx, word in enumerate(words):
        word = word.replace('\n', '')
        word = word.replace('\t', '')
        word_dict[idx] = word

    return word_dict


def inverse_word_dict(word_dict):
    iword_dict = {}
    for key, word in word_dict.items():
        iword_dict[word] = key
    return iword_dict


def load_embedding(embedding_path, iword_dict):
    embed_dict = {}
    for line in open(embedding_path):
        line = line.strip().split(' ')
        if line[0] not in iword_dict:
            continue
        else:
            key = iword_dict[line[0]]
            embed_dict[key] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (embedding_path, len(embed_dict)))
    return embed_dict


def embed_2_numpy(embed_dict, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros((len(embed_dict), feat_size), np.float32)
    for k in embed_dict:
        embed[k] = np.asarray(embed_dict[k])
    print('Generate numpy embed:', embed.shape)
    return embed


def sent_2_index(sent, word_dict, max_length, tokenizer="simple"):
    if tokenizer == "nltk":
        words = nltk_tokenizer(sent)
        word_indexes = []
        for idx, word in enumerate(words):
            if idx >= max_length:
                continue
            else:
                word_indexes.append(word_dict[word.lower()])
        return word_indexes
    # elif tokenizer=="simple":
    #     words = simple_tokenizer(sent)
    #     word_indexes = []
    #     for idx, word in enumerate(words):
    #         if idx >= max_length:
    #             continue
    #         else:
    #             word_indexes.append(word_dict[word.lower()])
    #     return word_indexes


def train_data_indexes(X, word_dict, h_max_length, s_max_length):
    X_indexes = []
    # nlp = StanfordCoreNLP(corenlp_path)
    print("start index words into intergers")
    for claim, pos, neg in X:
        claim_indexes = sent_2_index(claim, word_dict, h_max_length, tokenizer="nltk")
        pos_indexes = sent_2_index(pos, word_dict, s_max_length, tokenizer="nltk")
        neg_indexes = sent_2_index(neg, word_dict, s_max_length, tokenizer="nltk")
        X_indexes.append((claim_indexes, pos_indexes, neg_indexes))
    print('Training data size:', len(X_indexes))
    # nlp.close()
    return X_indexes


def test_data_indexes(devs, word_dict, h_max_length, s_max_length):
    devs_indexes = []
    # nlp = StanfordCoreNLP(corenlp_path)
    for dev in devs:
        sent_indexes = []
        print(dev)
        claim = dev[0][0]
        claim_index = sent_2_index(claim, word_dict, h_max_length, tokenizer="nltk")
        claim_indexes = [claim_index] * len(dev)
        for claim, sent in dev:
            sent_index = sent_2_index(sent, word_dict, s_max_length, tokenizer="nltk")
            sent_indexes.append(sent_index)
        assert len(sent_indexes) == len(claim_indexes)
        dev_indexes = list(zip(claim_indexes, sent_indexes))
        devs_indexes.append(dev_indexes)
    # nlp.close()
    return devs_indexes


def train_split(X, split_ratio=0.05):
    random.shuffle(X)
    split_point = int(len(X) * split_ratio)
    train_indexes = X[split_point:]
    dev_indexes = X[:split_point]
    print("Split for training:{}, split for testing:{}".format(len(train_indexes), len(dev_indexes)))
    return train_indexes, dev_indexes


def inputs_processing(base_path, db_filename, train_datapath, dev_datapath, embedding_path, h_max_length, s_max_length,
                      num_sample=1, k=5,
                      sampling="same_doc"):
    train_sample_path = "data/train_data/train_sample.s{}.k{}.{}.p".format(num_sample, k, sampling)
    dev_store_path = "data/train_data/dev.p"
    word_dict_path = "data/train_data/train.s_{}.h_{}.s{}.k{}.words.{}.p".format(s_max_length, h_max_length, num_sample,
                                                                                 k, sampling)
    train_sample_path = os.path.join(base_path, train_sample_path)
    dev_store_path = os.path.join(base_path, dev_store_path)
    print(dev_store_path)
    word_dict_path = os.path.join(base_path, word_dict_path)

    if os.path.exists(word_dict_path):
        with open(word_dict_path, "rb") as f:
            word_dict = pickle.load(f)
    else:
        X = train_data_loader(train_sample_path, db_filename, train_datapath, num_samples=num_sample, k=k,
                              sampling=sampling)
        devs, location_indexes = dev_data_loader(dev_store_path, db_filename, dev_datapath)
        train_words = get_train_words(X, h_max_length, s_max_length)
        dev_words = get_dev_words(devs, h_max_length, s_max_length)
        train_words.update(dev_words)
        word_dict = word_2_dict(train_words)
        with open(word_dict_path, "wb") as f:
            pickle.dump(word_dict, f)

    iword_dict = inverse_word_dict(word_dict)

    train_indexes_path = os.path.join(base_path,
                                      "data/train_data/train.s{}.k{}.h_{}.s_{}.{}.indexes.p".format(num_sample, k,
                                                                                                    h_max_length,
                                                                                                    s_max_length,
                                                                                                    sampling))
    if os.path.exists(train_indexes_path):
        with open(train_indexes_path, "rb") as f:
            X_indexes = pickle.load(f)
    else:
        X = train_data_loader(train_sample_path, db_filename, train_datapath, num_samples=num_sample, k=k)
        X_indexes = train_data_indexes(X, iword_dict, h_max_length, s_max_length)
        with open(train_indexes_path, "wb") as f:
            pickle.dump(X_indexes, f)

    embed_dict = load_embedding(embedding_path, iword_dict)
    # embed_words = set([word for word in embed_dict.keys()])
    # word_dict_words = set([word for word in iword_dict.keys()])
    # for word in word_dict_words:
    #     if word not in embed_words:
    #         print(word)
    embed_size = len(embed_dict[list(embed_dict.keys())[0]])
    print("embed_dict size {}".format(len(embed_dict)))
    _PAD_ = len(word_dict)
    word_dict[_PAD_] = '[PAD]'
    iword_dict['[PAD]'] = _PAD_
    embed_dict[_PAD_] = np.zeros(shape=(embed_size,), dtype=np.float32)
    init_embed = np.asarray(np.random.uniform(-0.25, 0.25, [len(word_dict), embed_size]), np.float32)
    embed = embed_2_numpy(embed_dict, embed=init_embed)
    print("embed shape :{}".format(embed.shape))
    X_train_indexes, X_dev_indexes = train_split(X_indexes)

    return X_train_indexes, X_dev_indexes, iword_dict, embed


def test_processing(base_path, dbfilename, test_data_path, test_store_path, h_max_length, s_max_length, iword_dict):
    dev_index_path = os.path.join(base_path,
                                  "data/train_data/dev.h_{}.s_{}.indexes.p".format(h_max_length, s_max_length))
    devs, location_indexes = dev_data_loader(test_store_path, dbfilename, test_data_path)
    if os.path.exists(dev_index_path):
        with open(dev_index_path, "rb") as f:
            devs_indexes = pickle.load(f)
    else:
        devs_indexes = test_data_indexes(devs, iword_dict, h_max_length, s_max_length)
        with open(dev_index_path, "wb") as f:
            pickle.dump(devs_indexes, f)

    return devs_indexes, location_indexes


def tfidf_test_processing(base_path, dbfilename, test_data_path, test_store_path, pro_extract_sents_path, h_max_length,
                          s_max_length, iword_dict):
    dev_index_path = os.path.join(base_path,
                                  "data/train_data/dev.h_{}.s_{}.tfidf.indexes.p".format(h_max_length, s_max_length))
    devs, location_indexes = dev_data_loader(test_store_path, dbfilename, test_data_path)
    if os.path.exists(dev_index_path):
        with open(dev_index_path, "rb") as f:
            devs_indexes = pickle.load(f)
    else:
        with open(pro_extract_sents_path, "r") as f:
            jlr = JSONLineReader()
            lines = jlr.process(f)

            inputs = []
            new_location_indexes = []
            for i, line in enumerate(lines):
                pro_extract_sents = []
                sent_index = []
                predict_sents = line['predicted_sentences']
                claim = line['claim']
                predict_sents_set = set([(doc_id, sent_num) for doc_id, sent_num in predict_sents])
                # print(predict_sents_set)
                for j, index in enumerate(location_indexes[i]):
                    if (index[0], index[1]) in predict_sents_set:
                        # print(devs[i][j])
                        # print(devs[i])
                        pro_extract_sents.append((claim, devs[i][j][1]))
                        sent_index.append((index[0], index[1]))
                inputs.append(pro_extract_sents)
                new_location_indexes.append(sent_index)
            devs_indexes = test_data_indexes(inputs, iword_dict, h_max_length, s_max_length)
    return devs_indexes, new_location_indexes
