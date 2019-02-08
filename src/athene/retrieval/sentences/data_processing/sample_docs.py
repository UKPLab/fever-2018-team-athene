import json
import os
import random
import re

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB

random.seed(38)


def sample_doc(claim, doc_ids, k=5):
    """
    random sample 5 documents to generate a dataset with golden documents in the predicted_pages
    :param claim:
    :param doc_ids:
    :param k:
    :return:
    """
    evidence_pages = set()
    for evidence in claim['all_evidence']:
        page = evidence[2]
        if page == None:
            continue
        elif page not in evidence_pages:
            evidence_pages.add(page)

    if len(evidence_pages) < k:
        samples = random.sample(doc_ids, k - len(evidence_pages))
        for sample in samples:
            evidence_pages.add(sample)
        if len(evidence_pages) < k:
            samples = random.sample(doc_ids, k - len(evidence_pages))
            for sample in samples:
                evidence_pages.add(sample)

    elif len(evidence_pages) >= k:

        samples = random.sample(evidence_pages, k)
        evidence_pages = set(samples)
    return evidence_pages


path = os.getcwd()
path = re.sub("/src.*", "", path)
db = FeverDocDB(os.path.join(path, "data/fever/fever.db"))
doc_ids = db.get_doc_ids()
doc_ids = doc_ids[1:]
jlr = JSONLineReader()
# with open(os.path.join(path, "data/fever-data/train.jsonl"), "r") as f:
#     with open(os.path.join(path, 'data/fever/train.p5.jsonl'), "w") as f2:
#         lines = f.readlines()
#         for line in lines:
#             js = json.loads(line)
#             pages = sample_doc(js,doc_ids,k=5)
#             js['predicted_pages'] = list(pages)
#             f2.write(json.dumps(js)+"\n")

with open(os.path.join(path, "data/fever-data/dev.jsonl"), "r") as f:
    with open(os.path.join(path, "data/fever/dev.p5.jsonl"), "w") as f2:
        lines = f.readlines()
        for line in lines:
            js = json.loads(line)
            pages = sample_doc(js, doc_ids, k=5)
            js['predicted_pages'] = list(pages)
            f2.write(json.dumps(js) + '\n')

# with open(os.path.join(path,'data/fever/train.p5.jsonl'),"r") as f:
#     lines = jlr.process(f)
#     print(len(lines))
#
#     for line in lines:
#         evidence_pages = set()
#         for evidence in line['predicted_pages']:
#             if evidence not in evidence_pages:
#                 evidence_pages.add(evidence)
#         if len(evidence_pages) < 5:
#             print(line['claim'])
