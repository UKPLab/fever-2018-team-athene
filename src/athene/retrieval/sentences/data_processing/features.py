from tqdm import tqdm

from drqa.retriever.tfidf_doc_ranker import TfidfDocRanker


def tfidf_transform(claim,sent,tfidf_path):

    tfidf = TfidfDocRanker(tfidf_path)
    tfidf_claim = tfidf.text2spvec(claim)
    tfidf_sent = tfidf.text2spvec(sent)

    print(tfidf_claim)
    print(tfidf_sent)
    print(len(tfidf_sent))
    print(len(tfidf_claim))


import os
path = os.getcwd()
path = path.replace("/src/data_processing","")
tfidf_transform("Sarah Michelle Gellar was in a movie.","Sarah Michelle Gellar (born April 14, 1977) is an American actress, producer, and entrepreneur.",tfidf_path=os.path.join(path,))