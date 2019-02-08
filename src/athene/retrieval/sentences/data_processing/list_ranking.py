import os
import pickle
import random

import numpy as np
from tqdm import tqdm

from athene.retrieval.sentences.data_processing.ranking_data import get_valid_texts, get_whole_evidence, nltk_tokenizer, \
    word_2_dict, inverse_word_dict, load_embedding, sent_2_index, embed_2_numpy
from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB

np.random.seed(100)
random.seed(100)


def train_sample(db_filename, lines, list_size=15):
    db = FeverDocDB(db_filename)

    claims = []
    list_sents = []
    labels = []
    count = 0

    for idx, line in tqdm(enumerate(lines)):

        if line['label'].upper() == "NOT ENOUGH INFO":
            continue

        claim = line['claim']
        claims.append(claim)
        sents = []
        label = []

        pos_set = set()
        neg_sents = []
        for evidence_group in line['evidence']:
            pos_sent = get_whole_evidence(evidence_group, db)
            if pos_sent in pos_set:
                continue
            pos_set.add(pos_sent)

        p_lines = []
        evidence_set = set([(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])

        pages = [page[0] for page in line['predicted_pages'] if page[0] is not None]
        for page, num in evidence_set:
            pages.append(page)
        pages = set(pages)
        for page in pages:
            doc_lines = db.get_doc_lines(page)
            p_lines.extend(get_valid_texts(doc_lines, page))
        for doc_line in p_lines:
            if not doc_line[0]:
                continue
            if (doc_line[1], doc_line[2]) not in evidence_set:
                neg_sents.append(doc_line[0])

        pos_set = list(pos_set)
        if len(pos_set) > 5:
            pos_set = random.sample(pos_set, 5)
        if len(neg_sents) < (list_size - len(pos_set)):

            count += 1
            continue
        else:
            samples = random.sample(neg_sents, list_size - len(pos_set))
            pos_indexes_sample = random.sample(range(list_size), len(pos_set))
            neg_index = 0
            pos_index = 0
            for i in range(list_size):
                if i in pos_indexes_sample:
                    sents.append(pos_set[pos_index])
                    label.append(1 / len(pos_set))
                    pos_index += 1
                else:
                    sents.append(samples[neg_index])
                    label.append(0.0)
                    neg_index += 1
            if idx % 1000 == 0:
                print(claim)
                print(sents)
                print(label)

        list_sents.append(sents)
        labels.append(label)
    print(count)
    return claims, list_sents, labels


def dev_processing(db_filename, lines):
    db = FeverDocDB(db_filename)
    claims = []
    list_sents = []
    labels = []

    for line in tqdm(lines):
        if line['label'].upper() == "NOT ENOUGH INFO":
            continue

        claims.append(line['claim'])
        sents = []
        label = []

        evidence_set = set([(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])
        pages = [page[0] for page in line['predicted_pages'] if page[0] is not None]
        for page, num in evidence_set:
            pages.append(page)
        pages = set(pages)

        p_lines = []
        for page in pages:
            doc_lines = db.get_doc_lines(page)
            p_lines.extend(get_valid_texts(doc_lines, page))
        for doc_line in p_lines:
            if not doc_line[0]:
                continue
            if (doc_line[1], doc_line[2]) in evidence_set:
                sents.append(doc_line[0])
                label.append(1)
            else:
                sents.append(doc_line[0])
                label.append(0)
        if len(claims) == 0 or len(list_sents) == 0 or len(labels) == 0:
            continue
        list_sents.append(sents)
        labels.append(label)
    return claims, list_sents, labels


def test_processing(db_filename, lines):
    db = FeverDocDB(db_filename)
    claims = []
    list_sents = []
    sents_indexes = []

    for line in tqdm(lines):
        # if line['label'].upper() == "NOT ENOUGH INFO":
        #     continue

        claims.append(line['claim'])
        sents = []
        sents_index = []

        evidence_set = set([(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])
        pages = set([page[0] for page in line['predicted_pages'] if page[0] is not None])
        if len(pages) == 0:
            pages.add("Michael_Hutchence")

        p_lines = []
        for page in pages:
            doc_lines = db.get_doc_lines(page)
            p_lines.extend(get_valid_texts(doc_lines, page))
        for doc_line in p_lines:
            if not doc_line[0]:
                continue
            if (doc_line[1], doc_line[2]) in evidence_set:
                sents.append(doc_line[0])
            else:
                sents.append(doc_line[0])
            sents_index.append((doc_line[1], doc_line[2]))
        list_sents.append(sents)
        sents_indexes.append(sents_index)
    return claims, list_sents, sents_indexes


def get_words(claims, list_sents, h_max_length, s_max_length):
    words = set()
    for claim in claims:
        for idx, word in enumerate(nltk_tokenizer(claim)):
            if idx >= h_max_length:
                break
            words.add(word.lower())

    for sents in list_sents:
        for sent in sents:
            for idx, word in enumerate(nltk_tokenizer(sent)):
                if idx >= s_max_length:
                    break
                words.add(word.lower())
    return words


def data_indexes(claims, list_sents, word_dict, h_max_length, s_max_length):
    claims_2_num = []
    lists_sents_2_num = []
    for i, claim in enumerate(claims):
        length = len(list_sents[i])
        index = sent_2_index(sent=claim, word_dict=word_dict, max_length=h_max_length, tokenizer="nltk")
        indexes = [index] * length
        claims_2_num.append(indexes)
    for sents in list_sents:
        sents_2_num = []
        for sent in sents:
            sents_2_num.append(sent_2_index(sent, word_dict, s_max_length, tokenizer="nltk"))
        lists_sents_2_num.append(sents_2_num)
    return claims_2_num, lists_sents_2_num


def train_dev_split(train_datapath, split_rate):
    with open(train_datapath, "r") as f:
        jlr = JSONLineReader()
        lines = jlr.process(f)
        random.shuffle(lines)

        dev_lines = lines[:int(len(lines) * split_rate)]
        train_lines = lines[int(len(lines) * split_rate):]
    return train_lines, dev_lines


def data_loader(sample_data_path, db_filename=None, lines=None, list_size=None, type="train"):
    if os.path.exists(sample_data_path):
        with open(sample_data_path, 'rb') as f:
            X = pickle.load(f)
            claims, list_sents, labels = zip(*X)
    else:
        if type == "train":
            claims, list_sents, labels = train_sample(db_filename, lines, list_size=list_size)
        elif type == "dev":
            claims, list_sents, labels = dev_processing(db_filename, lines)
        X = zip(claims, list_sents, labels)
        with open(sample_data_path, 'wb') as f:
            pickle.dump(X, f)
    return claims, list_sents, labels


def test_data_loader(save_path, db_filename=None, data_path=None):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            X = pickle.load(f)
            claims, list_sents, sents_indexes = zip(*X)
    else:
        with open(data_path, "rb") as f:
            jlr = JSONLineReader()
            lines = jlr.process(f)
        claims, list_sents, sents_indexes = test_processing(db_filename, lines)
        X = zip(claims, list_sents, sents_indexes)
        with open(save_path, 'wb') as f:
            pickle.dump(X, f)
    return claims, list_sents, sents_indexes


def indexes_loader(save_path, word_dict, sample_data_path, h_max_length, s_max_length):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            X = pickle.load(f)
            claims_indexes, list_sents_indexes, labels = zip(*X)
    else:
        claims, list_sents, labels = data_loader(sample_data_path)
        claims_indexes, list_sents_indexes = data_indexes(claims, list_sents, word_dict, h_max_length, s_max_length)
        with open(save_path, "wb") as f:
            X = zip(claims_indexes, list_sents_indexes, labels)
            pickle.dump(X, f)
    return claims_indexes, list_sents_indexes, labels


def test_indexes_loader(save_path, word_dict, test_datapath, h_max_length, s_max_length):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            X = pickle.load(f)
            claims_indexes, list_sents_indexes, location_indexes = zip(*X)
    else:
        claims, list_sents, location_indexes = test_data_loader(test_datapath)
        claims_indexes, list_sents_indexes = data_indexes(claims, list_sents, word_dict, h_max_length, s_max_length)
        with open(save_path, "wb") as f:
            X = zip(claims_indexes, list_sents_indexes, location_indexes)
            pickle.dump(X, f)
    return claims_indexes, list_sents_indexes, location_indexes


def inputs_processing(base_path, db_filename, train_datapath, dev_datapath, embedding_path, h_max_length, s_max_length,
                      list_size=15):
    train_lines, dev_lines = train_dev_split(train_datapath, split_rate=0.1)

    train_sample_path = "data/train_data/train_sample.list{}.p".format(list_size)
    dev_store_path = "data/train_data/dev.list{}.p".format(list_size)
    test_store_path = "data/train_data/test.p"
    word_dict_path = "data/train_data/train.s_{}.h_{}.list{}.words.p".format(s_max_length, h_max_length, list_size)
    train_sample_path = os.path.join(base_path, train_sample_path)
    dev_store_path = os.path.join(base_path, dev_store_path)
    test_store_path = os.path.join(base_path, test_store_path)
    word_dict_path = os.path.join(base_path, word_dict_path)

    if os.path.exists(word_dict_path):
        with open(word_dict_path, "rb") as f:
            word_dict = pickle.load(f)
        print('warm' in word_dict)
    else:
        train_claims, train_list_sents, train_labels = data_loader(train_sample_path, db_filename, train_lines,
                                                                   list_size, type="train")
        dev_claims, dev_list_sents, dev_labels = data_loader(dev_store_path, db_filename, dev_lines, list_size,
                                                             type="dev")
        test_claims, test_list_sents, test_sents_indexes = test_data_loader(test_store_path, db_filename, dev_datapath)
        train_words = get_words(train_claims, train_list_sents, h_max_length, s_max_length)
        dev_words = get_words(dev_claims, dev_list_sents, h_max_length, s_max_length)
        test_words = get_words(test_claims, test_list_sents, h_max_length, s_max_length)
        train_words.update(dev_words)
        train_words.update(test_words)
        print("get all words, words size {}".format(train_words))
        word_dict = word_2_dict(train_words)
        with open(word_dict_path, "wb") as f:
            pickle.dump(word_dict, f)

    iword_dict = inverse_word_dict(word_dict)

    train_indexes_path = os.path.join(base_path,
                                      "data/train_data/train.list{}.h_{}.s_{}.indexes.p".format(list_size, h_max_length,
                                                                                                s_max_length))
    dev_indexes_path = os.path.join(base_path,
                                    "data/train_data/dev.list{}.h_{}.s_{}.indexes.p".format(list_size, h_max_length,
                                                                                            s_max_length))

    print("train_data_indexing")
    train_claims, train_list_sents, train_labels = indexes_loader(train_indexes_path, iword_dict, train_sample_path,
                                                                  h_max_length, s_max_length)
    print("dev_data_indexing")
    dev_claims, dev_list_sents, dev_labels = indexes_loader(dev_indexes_path, iword_dict, dev_store_path, h_max_length,
                                                            s_max_length)

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

    return train_claims, train_list_sents, train_labels, dev_claims, dev_list_sents, dev_labels, iword_dict, embed


def test_input(base_path, dbfilename, test_data_path, test_store_path, h_max_length, s_max_length, list_size,
               word_dict):
    dev_index_path = os.path.join(base_path,
                                  "data/train_data/dev.h_{}.s_{}.list_{}.indexes.p".format(h_max_length, s_max_length,
                                                                                           list_size))
    test_claims, test_list_sents, test_location_indexes = test_indexes_loader(dev_index_path, word_dict,
                                                                              test_store_path, h_max_length,
                                                                              s_max_length)

    return test_claims, test_list_sents, test_location_indexes
