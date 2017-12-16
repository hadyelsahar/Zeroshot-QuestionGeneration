
import random
import pickle
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

class SELECT:

    def __init__(self, mode="pred"):
        """
        :param mode: subtype, pred, or objtype
               if pred : select objtype
               if subtype of objtype: select pred
        """

        self.mode = mode

    def train(self, x, y, textual_evidence=None):

        self.train_x = x
        self.train_y = y

        if self.mode == "pred":
            self.train_data = dict(zip([i[2] for i in x], y))
        else:
            self.train_data = dict(zip([i[1] for i in x], y))

    def test(self, x, textual_evidence=None):

        if self.mode == "pred":
            x = [i[2] for i in x]
        else:
            x = [i[1] for i in x]

        y = []
        for i in x:
            if i in self.train_data:
                y.append(self.train_data[i])
            else:
                r = random.randint(0, len(self.train_data.values())-1)
                y.append(self.train_data.values()[r])

        return y

class IR:

    def __init__(self, TEXT=False):
        """

        :param TEXT: boolean to determine doing IR over text or IR over KB embeddings
        """

        self.K = 5
        self.RADIUS = 0.4
        self.N_COMPONENTS = 200
        self.TEXT = TEXT

    def train(self, x, y, textual_evidence=None):

        if self.TEXT:
            x = textual_evidence

            x = [str(i) for i in x]  # make sure every word is string

        self.train_x = x
        self.train_y = y

        # VECTORIZATION
        print("vectorization..")
        # X = np.concatenate([train['triples_prep'], test['triples_prep']])

        self.count_vect = CountVectorizer().fit(x)
        x = self.count_vect.transform(x)

        self.tf_transformer = TfidfTransformer().fit(x)
        x = self.tf_transformer.transform(x)

        self.svd = TruncatedSVD(n_components=self.N_COMPONENTS).fit(x)
        x = self.svd.transform(x)

        # CLUSTERING
        print("clustering..")
        self.neigh = NearestNeighbors(self.K, self.RADIUS)
        self.neigh.fit(x)  # clustering only training set

    def test(self, x, textual_evidence=None):

        if self.TEXT:
            x = textual_evidence

        x = [str(i) for i in x]  # make sure every word is string

        x = self.count_vect.transform(x)
        x = self.tf_transformer.transform(x)
        x = self.svd.transform(x)

        dist, id = self.neigh.kneighbors(x, 1)
        y = self.train_y[id[0][0]]

        return [y]


class RTransE:

    def __init__(self, transe_entities_path, transe_predicates_path):

        self.K = 5
        self.RADIUS = 0.4
        self.N_COMPONENTS = 200
        self.Etranse = pickle.load(open(transe_entities_path))
        self.Ptranse = pickle.load(open(transe_predicates_path))

    def train(self, x, y, textual_evidence=None):

        x = np.array([np.concatenate((self.Etranse[i[0]], self.Ptranse[i[1]], self.Etranse[i[2]])) for i in x])

        self.train_x = x
        self.train_y = y

        # CLUSTERING
        print("clustering..")
        self.neigh = NearestNeighbors(self.K, self.RADIUS)
        self.neigh.fit(x)  # clustering only training set

    def test(self, x, textual_evidence=None):

        x = np.array([np.concatenate((self.Etranse[i[0]], self.Ptranse[i[1]], self.Etranse[i[2]])) for i in x])

        dist, id = self.neigh.kneighbors(x, 1)
        y = self.train_y[id[0][0]]

        return [y]
