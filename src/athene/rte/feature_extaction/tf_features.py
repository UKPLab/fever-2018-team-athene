from utils import dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from csv import DictReader
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

label_ref = {'agree': 0, 'refute': 1, 'nostance': 2}
label_ref_rev = {0: 'agree', 1: 'refute', 2: 'nostance'}

stop_words = stopwords.words('english')


class FNCData:
    """
    Define class for Fake News Challenge data
    """

    def __init__(self, file_instances, file_bodies):

        # Load data
        self.instances = self.read(file_instances)
        bodies = self.read(file_bodies)
        self.heads = {}
        self.bodies = {}

        # Process instances
        for instance in self.instances:
            if instance['Claim'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Claim']] = head_id
            instance['Body ID'] = int(instance['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['Snippets']

    def read(self, filename):

        """
        Read Fake News Challenge data from CSV file
        Args:
            filename: str, filename + extension
        Returns:
            rows: list, of dict per instance
        """

        # Initialise
        rows = []

        # Process file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows


# Initialise
heads = []
heads_track = {}
bodies = []
bodies_track = {}
body_ids = []
id_ref = {}
train_set = []
train_stances = []
cos_track = {}
test_heads = []
test_heads_track = {}
test_bodies = []
test_bodies_track = {}
test_body_ids = []
head_tfidf_track = {}
body_tfidf_track = {}

file_train_claims = "splits/train_claims.csv"
file_train_evidences = "splits/train_evidences.csv"
file_dev_claims = "splits/dev_claims.csv"
file_dev_evidences = "splits/dev_evidences.csv"
train = FNCData(file_train_claims, file_train_evidences)
test = FNCData(file_dev_claims, file_dev_evidences)

for stance in train.instances:
    print(stance["Stance"])
    print(stance['Claim'])
    body_id = stance['Body ID']
    print(train.bodies[body_id])
    print("=================================================")
# Identify unique heads and bodie
for instance in train.instances:
    head = instance['Claim']
    body_id = instance['Body ID']
    if head not in heads_track:
        heads.append(head)
        heads_track[head] = 1
    if body_id not in bodies_track:
        bodies.append(train.bodies[body_id])
        bodies_track[body_id] = 1
        body_ids.append(body_id)

for instance in test.instances:
    head = instance['Claim']
    body_id = instance['Body ID']
    if head not in test_heads_track:
        test_heads.append(head)
        test_heads_track[head] = 1
    if body_id not in test_bodies_track:
        test_bodies.append(test.bodies[body_id])
        test_bodies_track[body_id] = 1
        test_body_ids.append(body_id)

    # Create reference dictionary
for i, elem in enumerate(heads + body_ids):
    id_ref[elem] = i

# Create vectorizers and BOW and TF arrays for train set
bow_vectorizer = CountVectorizer(stop_words=stop_words)
bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words). \
    fit(heads + bodies + test_heads + test_bodies)  # Train and test sets

# Process train set
for instance in train.instances:
    head = instance['Claim']
    body_id = instance['Body ID']
    head_tf = tfreq[id_ref[head]].reshape(1, -1)
    body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
    if head not in head_tfidf_track:
        head_tfidf = tfidf_vectorizer.transform([head]).toarray()
        head_tfidf_track[head] = head_tfidf
    else:
        head_tfidf = head_tfidf_track[head]
    if body_id not in body_tfidf_track:
        body_tfidf = tfidf_vectorizer.transform([train.bodies[body_id]]).toarray()
        body_tfidf_track[body_id] = body_tfidf
    else:
        body_tfidf = body_tfidf_track[body_id]
    if (head, body_id) not in cos_track:
        tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
        cos_track[(head, body_id)] = tfidf_cos
    else:
        tfidf_cos = cos_track[(head, body_id)]
    feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
    train_set.append(feat_vec)
    train_stances.append(label_ref[instance['Stance']])

print(len(train_set))
print(len(train_stances))
# Initialise
test_set = []
heads_track = {}
bodies_track = {}
cos_track = {}
test_stances = []
# Process test set
for instance in test.instances:
    head = instance['Claim']
    body_id = instance['Body ID']
    if head not in heads_track:
        head_bow = bow_vectorizer.transform([head]).toarray()
        head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
        head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
        heads_track[head] = (head_tf, head_tfidf)
    else:
        head_tf = heads_track[head][0]
        head_tfidf = heads_track[head][1]
    if body_id not in bodies_track:
        body_bow = bow_vectorizer.transform([test.bodies[body_id]]).toarray()
        body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
        body_tfidf = tfidf_vectorizer.transform([test.bodies[body_id]]).toarray().reshape(1, -1)
        bodies_track[body_id] = (body_tf, body_tfidf)
    else:
        body_tf = bodies_track[body_id][0]
        body_tfidf = bodies_track[body_id][1]
    if (head, body_id) not in cos_track:
        tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
        cos_track[(head, body_id)] = tfidf_cos
    else:
        tfidf_cos = cos_track[(head, body_id)]
    feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
    test_set.append(feat_vec)
    test_stances.append(label_ref[instance['Stance']])

clf = MultinomialNB()
clf.fit(train_set, train_stances)
score = clf.score(test_set, test_stances)
print(score)
