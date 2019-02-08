import random

from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB


def label_sents(db_path, data_path, type="train"):
    """
    This function is to label all sentences in the evidence set to 1 and not in evidence set to 0 for training data
    :param db_path:
    :param data_path:
    :param type:
    :return:
    """

    db = FeverDocDB(db_path)
    jsr = JSONLineReader()
    claims = []
    related_pages_sents = []
    pages_sents_indexes = []
    y = []
    with open(data_path, "r") as f:
        lines = jsr.process(f)
        count = 0
        for line in tqdm(lines):
            if line['label'] == "NOT ENOUGH INFO" and type == "train":
                continue
            p_lines = []
            valid_lines = []
            line_labels = []
            sents_idnexes = []
            claim = line['claim']
            evidences = line['evidence']
            evidence_set = set()
            pages_list = []
            for evidence in evidences:
                for sent in evidence:
                    evidence_set.add((sent[2], sent[3]))
                    pages_list.append(sent[2])
            # predicted_pages = line['predicted_pages']
            predicted_pages = [page[0] for page in line['predicted_pages']]
            predicted_pages = predicted_pages + pages_list
            predicted_pages = set(predicted_pages)
            if len(predicted_pages) > 5:
                count += 1
            claims.append(claim)
            for page in predicted_pages:
                doc_lines = db.get_doc_lines(page)
                if not doc_lines:
                    # print(page)
                    continue
                doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                             doc_lines.split("\n")]
                p_lines.extend(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))

            for doc_line in p_lines:
                # ignore empty sentences
                if not doc_line[0]:
                    continue
                else:
                    # print(doc_line[0])
                    sents_idnexes.append((doc_line[1], doc_line[2]))
                    valid_lines.append(doc_line[0])
                    is_added = False
                    for sent in evidence_set:
                        if sent[0] == doc_line[1] and sent[1] == doc_line[2]:
                            line_labels.append(1)
                            is_added = True
                            break
                    if is_added != True:
                        line_labels.append(0)
            # print(len(p_lines))
            # print(len(line_labels))
            # print(len(valid_lines))
            assert len(line_labels) == len(valid_lines) == len(sents_idnexes)
            related_pages_sents.append(valid_lines)
            pages_sents_indexes.append(sents_idnexes)
            y.append(line_labels)
    print(count)
    return claims, related_pages_sents, pages_sents_indexes, y


def sample_4_ranking(db_path, data_path, type="train", num_neg=3, seed=55):
    """
    sample a set of negative sentecnes and combine positive sentence to form the training data
    :param db_path:
    :param data_path:
    :param type:
    :param num_neg:
    :param seed:
    :return:
    """

    random.seed(seed)
    if type == "train":
        claims, related_pages_sents, _, y = label_sents(db_path, data_path, type="train")

        train_triplets = []
        for i, claim in tqdm(enumerate(claims)):
            neg_sents = [j for j, label in enumerate(y[i]) if label != 1]
            for idx, label in enumerate(y[i]):
                if label == 1:
                    pos_sent = related_pages_sents[i][idx]
                    samples = random.sample(neg_sents, num_neg)
                    sampled_neg_sents = []
                    for index in samples:
                        sampled_neg_sents.append(related_pages_sents[i][index])
                    triplet = (claim, pos_sent, sampled_neg_sents)
                    train_triplets.append(triplet)
        return train_triplets

    elif type == "dev" or type == "test":

        """
        For dev or test set, use claim and sentence pairs to get scores of each pair
        """
        db = FeverDocDB(db_path)
        jsr = JSONLineReader()
        with open(data_path, "r") as f:
            lines = jsr.process(f)

            dev_examples = []
            pages_sents_indexes = []
            for line in tqdm(lines):
                p_lines = []
                feed_tuples = []
                sents_indexes = []
                claim = line['claim']
                for page in line['predicted_pages']:
                    doc_lines = db.get_doc_lines(page)
                    if not doc_lines:
                        # print(page)
                        continue
                    doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                                 doc_lines.split("\n")]
                    p_lines.extend(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))

                for doc_line in p_lines:
                    if not doc_line[0]:
                        continue
                    else:
                        # print(doc_line[0])
                        sents_indexes.append((doc_line[1], doc_line[2]))
                        feed_tuples.append((claim, doc_line[0]))

                dev_examples.append(feed_tuples)
                pages_sents_indexes.append(sents_indexes)

        return dev_examples, pages_sents_indexes


def cos_train(db_filepath, dataset_path):
    """
    Use the cosine similarity score to rank (claim,sentence) pair in the dev set
    don't need training data
    :param db_filepath:
    :param dataset_path:
    :return:
    """

    db = FeverDocDB(db_filepath)
    jlr = JSONLineReader()

    X = []
    y = []
    with open(dataset_path, "r") as f:
        lines = jlr.process(f)

        for line in tqdm(lines):
            if line['label'] == "NOT ENOUGH INFO":
                continue
            p_lines = []
            claim = line['claim']
            evidence_set = set([(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])
            pages = set()
            pages.update(evidence[0] for evidence in line['predicted_pages'])
            pages.update(evidence[0] for evidence in evidence_set)
            for page in pages:
                doc_lines = db.get_doc_lines(page)
                if not doc_lines:
                    continue
                doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                             doc_lines.split("\n")]
                p_lines.extend(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))
            for doc_line in p_lines:
                if not doc_line[0]:
                    continue
                else:
                    X.append((claim, doc_line[0]))
                    if (doc_line[1], doc_line[2]) in evidence_set:
                        y.append(1)
                    else:
                        y.append(0)
    return X, y


def prepare_ranking(db_filename, datapath, k=10, num_sample=3):
    """

    :param db_filename:
    :param datapath:
    :param k:
    :param num_sample:
    :return:
    """

    db = FeverDocDB(db_filename)
    jlr = JSONLineReader()

    X = []
    with open(datapath, "r") as f:
        lines = jlr.process(f)

        for line in tqdm(lines):
            if line['label'].upper() == "NOT ENOUGH INFO":
                continue
            p_lines = []
            pos_sents = []
            neg_sents = []
            claim = line['claim']
            evidence_set = set([(evidence[2], evidence[3]) for evidences in line['evidence'] for evidence in evidences])
            sampled_sents_idx = [(id, number) for id, number in line['predicted_sentences']]
            sampled_sents_idx = [index for index in sampled_sents_idx if index not in evidence_set]
            if k:
                sampled_sents_idx = sampled_sents_idx[:k]
            pages = set()
            pages.update(evidence[0] for evidence in line['predicted_pages'])
            pages.update(evidence[0] for evidence in evidence_set)
            for page in pages:
                doc_lines = db.get_doc_lines(page)
                if not doc_lines:
                    continue
                doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
                             doc_lines.split("\n")]
                p_lines.extend(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))
            for doc_line in p_lines:
                if not doc_line[0]:
                    continue
                elif (doc_line[1], doc_line[2]) in sampled_sents_idx:
                    neg_sents.append(doc_line[0])
                elif (doc_line[1], doc_line[2]) in evidence_set:
                    pos_sents.append(doc_line[0])
            # print(line)
            # print(sampled_sents_idx)
            # print(neg_sents)
            if len(sampled_sents_idx) < num_sample:
                continue
            for sent in pos_sents:
                neg_samples = random.sample(neg_sents, num_sample)
                triplet = (claim, sent, neg_samples)
                X.append(triplet)

    return X

# if __name__ == "__main__":
#     path = os.getcwd()
#     path = re.sub("src.*","",path)
#
#     db_filename = os.path.join(path,"data/fever/fever.db")
#     data_path = os.path.join(path,"data/fever/train.p5.jsonl")
#
#     claims,related_pages_sents,y = label_sents(db_filename,data_path)
#     assert len(claims) == len(related_pages_sents) == len(y)
