import os
import math

import pandas as pd
import numpy as np

class Data:
    """
    class acts as a convenient data feeder for few shots learning

    """

    def __init__(self, datapath=None, seed=3, remove_unk=False):

        self.remove_unk = remove_unk

        if datapath is None:
            datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./preprocessed")

        np.random.seed(seed)

        # loading vocab
        def return_vocab(filename):

            keys = [l.strip() for l in open(filename).readlines()]
            values = range(0, len(keys))
            return dict(zip(keys, values))

        def return_inv_vocab(filename):

            keys = [l.strip() for l in open(filename).readlines()]
            values = range(0, len(keys))
            return dict(zip(values, keys))

        self.entityvocab = return_vocab(os.path.join(datapath, "entity.vocab"))
        self.propertyvocab = return_vocab(os.path.join(datapath, "property.vocab"))
        self.wordvocab = return_vocab(os.path.join(datapath, "word.vocab"))

        self.inv_entityvocab = return_inv_vocab(os.path.join(datapath, "entity.vocab"))
        self.inv_propertyvocab = return_inv_vocab(os.path.join(datapath, "property.vocab"))
        self.inv_wordvocab = return_inv_vocab(os.path.join(datapath, "word.vocab"))

        # loading data files names
        self.datafile = {"train":os.path.join(datapath, "train.ids"),
                         "valid": os.path.join(datapath, "valid.ids"),
                         "test": os.path.join(datapath, "valid.ids")
                         }

        self.data = {}

    def read_data(self, mode):
        # modes = ["train", "test", "valid"]
        data = []

        f = self.datafile[mode]
        x = pd.read_csv(f, names=["sub", "pred", "obj", "question", "subtype", "objtype", "dep", "direction", "placeholder_dict"])

        if self.remove_unk:
            unkdep = self.wordvocab["_UNK_DEP_"] if "_UNK_DEP_" in self.wordvocab else None
            x = x[x.dep != unkdep]
            x = x[x.apply(lambda i: str(self.wordvocab["_PLACEHOLDER_SUB_"]) in i['question'].split(), axis=1)]

        x.reset_index(inplace=True)

        tmp = [[], [], [], []]
        for l in x.iterrows():

            tmp[0].append([int(i) for i in l[1]['question'].split()])
            tmp[1].append([int(i) for i in l[1]['subtype'].split()])
            tmp[2].append([int(i) for i in l[1]['objtype'].split()])
            tmp[3].append([int(i) for i in l[1]['dep'].split()])

        x['question'] = tmp[0]
        x['subtype'] = tmp[1]
        x['objtype'] = tmp[2]
        x['dep'] = tmp[3]

        x['question_length'] = x.apply(lambda l: len(l['question']), axis=1)
        x['subtype_length'] = x.apply(lambda l: len(l['subtype']), axis=1)
        x['objtype_length'] = x.apply(lambda l: len(l['objtype']), axis=1)
        x['dep_length'] = x.apply(lambda l: len(l['dep']), axis=1)
        x['triple'] = x.apply(lambda l: [l['sub'], l['pred'], l['obj']], axis=1)

        return x

    def datafeed(self, mode, config, shuffle=True):
        """

        :param mode: train, valid, test
        :param config: config object
        :param shot_percentage: float between 0 and 1 indicating the percentage of the training data taken into consideration
        :param min_count: int indicating the minimum count of the predicates of the examples being taken in to consideration
        :param shuffle: whether to shuffle the training data or not
        :param kfold: a number between 1 and 10
        :return:
        """

        x = self.read_data(mode)
        self.data[mode] = x

        dataids = x.index

        if shuffle:
            np.random.shuffle(dataids)

        return self.yield_datafeed(mode, dataids, x, config)


    def yield_datafeed(self, mode, dataids, x, config):
        """
        given a dataframe and selected ids and a mode yield data for experiments
        :param mode:
        :param dataids:
        :param x:
        :param config:
        :return:
        """

        if mode == "train":

            for epoch in range(config.MAX_EPOCHS):

                def chunks(l, n):
                    """Yield successive n-sized chunks from l."""
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                for bn, batchids in enumerate(chunks(dataids, config.BATCH_SIZE)):

                    batch = x.iloc[batchids]

                    max_length = max([batch['subtype_length'].values.max(),
                         batch['objtype_length'].values.max(),
                         batch['dep_length'].values.max(),
                         ])

                    yield (
                        np.array([i for i in batch['triple'].values]),
                        self.pad(batch['subtype'].values, max_length=max_length),
                        batch['subtype_length'].values,
                        self.pad(batch['objtype'].values, max_length=max_length),
                        batch['objtype_length'].values,
                        self.pad(batch['dep'].values, max_length=max_length),
                        batch['dep_length'].values,
                        self.pad(batch['question'].values),
                        batch['question_length'].values,
                        batch['direction'].values,
                        {"epoch": epoch, "batch_id": bn, "ids": batchids, "placeholder_dict":[eval(i) for i in batch["placeholder_dict"].values]}  # meta info
                    )

        if mode == "test" or mode == "valid":
            # in case of test of validation batch of size 1 and no shuffle
            # takes longer computation time but allows variable lengths

            for id in dataids:

                batch = x.iloc[[id]]

                max_length = max([batch['subtype_length'].values.max(),
                                  batch['objtype_length'].values.max(),
                                  batch['dep_length'].values.max(),
                                  ])

                yield (
                    np.array([i for i in batch['triple']]),
                    self.pad(batch['subtype'], max_length=max_length),
                    batch['subtype_length'].values,
                    self.pad(batch['objtype'], max_length=max_length),
                    batch['objtype_length'].values,
                    self.pad(batch['dep'].values, max_length=max_length),
                    batch['dep_length'].values,
                    self.pad(batch['question'].values),
                    batch['question_length'].values,
                    batch['direction'].values,
                    {"ids": id, "placeholder_dict": [eval(i) for i in batch["placeholder_dict"].values]}  # meta info
                )

    def pad(self, x, pad_char=0, max_length=None):
        """
        helper function to add padding to a batch
        :param x: array of arrays of variable length
        :return:  x with padding of max length
        """

        if max_length is None:
            max_length = max([len(i) for i in x])

        y = np.ones([len(x), max_length]) * pad_char

        for c, i in enumerate(x):
            y[c, :len(i)] = i

        return y

class FewShotsDataFeeder:
    """
    class acts as a convenient data feeder for few shots learning

    """

    def __init__(self, datapath=None, seed=3, train_percent=0.7, test_percent=0.2, remove_unk=False):

        assert 0 < train_percent + test_percent <= 1

        self.train_percent = train_percent
        self.test_percent = test_percent
        self.valid_percent = 1 - (train_percent+test_percent)
        self.remove_unk = remove_unk

        if datapath is None:
            datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./preprocessed")

        np.random.seed(seed)

        # loading vocab
        def return_vocab(filename):

            keys = [l.strip() for l in open(filename).readlines()]
            values = range(0, len(keys))
            return dict(zip(keys, values))

        def return_inv_vocab(filename):

            keys = [l.strip() for l in open(filename).readlines()]
            values = range(0, len(keys))
            return dict(zip(values, keys))

        self.entityvocab = return_vocab(os.path.join(datapath, "entity.vocab"))
        self.propertyvocab = return_vocab(os.path.join(datapath, "property.vocab"))
        self.wordvocab = return_vocab(os.path.join(datapath, "word.vocab"))

        self.inv_entityvocab = return_inv_vocab(os.path.join(datapath, "entity.vocab"))
        self.inv_propertyvocab = return_inv_vocab(os.path.join(datapath, "property.vocab"))
        self.inv_wordvocab = return_inv_vocab(os.path.join(datapath, "word.vocab"))

        # loading data files names
        self.datafile = {"train":os.path.join(datapath, "train.ids"),
                         "valid": os.path.join(datapath, "valid.ids"),
                         "test": os.path.join(datapath, "valid.ids")
                         }

        self.data = {}

    def read_data(self):
        modes = ["train", "test", "valid"]
        data = []
        for m in modes:
            f = self.datafile[m]
            data.append(pd.read_csv(f, names=["sub", "pred", "obj", "question", "subtype", "objtype", "dep", "direction", "placeholder_dict"]))

        x = pd.concat(data)

        if self.remove_unk:
            unkdep = self.wordvocab["_UNK_DEP_"] if "_UNK_DEP_" in self.wordvocab else None
            x = x[x.dep != unkdep]
            x = x[x.apply(lambda i: str(self.wordvocab["_PLACEHOLDER_SUB_"]) in i['question'].split(), axis=1)]

        x.reset_index(inplace=True)

        tmp = [[], [], [], []]
        for l in x.iterrows():

            tmp[0].append([int(i) for i in l[1]['question'].split()])
            tmp[1].append([int(i) for i in l[1]['subtype'].split()])
            tmp[2].append([int(i) for i in l[1]['objtype'].split()])
            tmp[3].append([int(i) for i in l[1]['dep'].split()])

        x['question'] = tmp[0]
        x['subtype'] = tmp[1]
        x['objtype'] = tmp[2]
        x['dep'] = tmp[3]

        x['question_length'] = x.apply(lambda l: len(l['question']), axis=1)
        x['subtype_length'] = x.apply(lambda l: len(l['subtype']), axis=1)
        x['objtype_length'] = x.apply(lambda l: len(l['objtype']), axis=1)
        x['dep_length'] = x.apply(lambda l: len(l['dep']), axis=1)
        x['triple'] = x.apply(lambda l: [l['sub'], l['pred'], l['obj']], axis=1)

        return x

    def filter_data(self, x, mode, shot_percentage, min_count):

        # removing predicates with less than min_count examples in the labeled set
        x = x.groupby("pred").filter(lambda x: len(x) >= min_count)
        ids = x.groupby("pred").indices

        keep_ids = np.array([], dtype=np.int)
        if mode == "train":
            for v in ids.values():
                start = 0
                end = int(math.ceil(len(v) * shot_percentage * self.train_percent))
                keep_ids = np.append(keep_ids, v[start:end])

        elif mode == "test":
            for v in ids.values():
                start = int(math.ceil(len(v) * shot_percentage * self.train_percent))
                end = int(start + math.ceil(len(v) * self.test_percent))
                keep_ids = np.append(keep_ids, v[start:end])

        elif mode == "valid":

            for v in ids.values():
                start = int(math.ceil(len(v) * shot_percentage * self.train_percent) + math.ceil(len(v) * self.test_percent))
                end = int(start + math.ceil(len(v) * self.valid_percent))
                keep_ids = np.append(keep_ids, v[start:end])

        return keep_ids, x

    def datafeed(self, mode, config, shot_percentage=1, min_count=10, shuffle=True):
        """

        :param mode: train, valid, test
        :param config: config object
        :param shot_percentage: float between 0 and 1 indicating the percentage of the training data taken into consideration
        :param min_count: int indicating the minimum count of the predicates of the examples being taken in to consideration
        :param shuffle: whether to shuffle the training data or not
        :param kfold: a number between 1 and 10
        :return:
        """

        x = self.read_data()
        self.data[mode] = x

        dataids, x = self.filter_data(x, mode, shot_percentage, min_count)
        dataids = [i for i in dataids if i in x.index]

        if shuffle:
            np.random.shuffle(dataids)

        return self.yield_datafeed(mode, dataids, x, config)


    def yield_datafeed(self, mode, dataids, x, config):
        """
        given a dataframe and selected ids and a mode yield data for experiments
        :param mode:
        :param dataids:
        :param x:
        :param config:
        :return:
        """

        if mode == "train":

            for epoch in range(config.MAX_EPOCHS):

                def chunks(l, n):
                    """Yield successive n-sized chunks from l."""
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                for bn, batchids in enumerate(chunks(dataids, config.BATCH_SIZE)):

                    batch = x.iloc[batchids]

                    max_length = max([batch['subtype_length'].values.max(),
                         batch['objtype_length'].values.max(),
                         batch['dep_length'].values.max(),
                         ])

                    yield (
                        np.array([i for i in batch['triple'].values]),
                        self.pad(batch['subtype'].values, max_length=max_length),
                        batch['subtype_length'].values,
                        self.pad(batch['objtype'].values, max_length=max_length),
                        batch['objtype_length'].values,
                        self.pad(batch['dep'].values, max_length=max_length),
                        batch['dep_length'].values,
                        self.pad(batch['question'].values),
                        batch['question_length'].values,
                        batch['direction'].values,
                        {"epoch": epoch, "batch_id": bn, "ids": batchids, "placeholder_dict":[eval(i) for i in batch["placeholder_dict"].values]}  # meta info
                    )

        if mode == "test" or mode == "valid":
            # in case of test of validation batch of size 1 and no shuffle
            # takes longer computation time but allows variable lengths

            for id in dataids:

                batch = x.iloc[[id]]

                max_length = max([batch['subtype_length'].values.max(),
                                  batch['objtype_length'].values.max(),
                                  batch['dep_length'].values.max(),
                                  ])

                yield (
                    np.array([i for i in batch['triple']]),
                    self.pad(batch['subtype'], max_length=max_length),
                    batch['subtype_length'].values,
                    self.pad(batch['objtype'], max_length=max_length),
                    batch['objtype_length'].values,
                    self.pad(batch['dep'].values, max_length=max_length),
                    batch['dep_length'].values,
                    self.pad(batch['question'].values),
                    batch['question_length'].values,
                    batch['direction'].values,
                    {"ids": id, "placeholder_dict": [eval(i) for i in batch["placeholder_dict"].values]}  # meta info
                )

    def pad(self, x, pad_char=0, max_length=None):
        """
        helper function to add padding to a batch
        :param x: array of arrays of variable length
        :return:  x with padding of max length
        """

        if max_length is None:
            max_length = max([len(i) for i in x])

        y = np.ones([len(x), max_length]) * pad_char

        for c, i in enumerate(x):
            y[c, :len(i)] = i

        return y


class ZeroShotsDataFeeder(FewShotsDataFeeder):

    def filter_data(self, x, mode, criteria="pred", min_count=10, kfold=10, cv=0):
        """

        :param x:
        :param mode:
        :param criteria:
        :param min_count:
        :param kfold:
        :param cv:
        :return:
        """

        # removing predicates with less than min_count examples in the labeled set
        criteria_hash = criteria + "_hash"
        x[criteria_hash] = x.apply(lambda a: str(a[criteria]), axis=1)  # make criteria hashable
        x = x.groupby(criteria_hash).filter(lambda i: len(i) >= min_count)
        ids = x.groupby(criteria_hash).indices
        ids = sorted(ids.items(), key=lambda a: len(a[1]), reverse=True)

        keep_ids = np.array([], dtype=np.int)

        if mode == "train":

            start = cv
            pos = [(i + start) % kfold for i in range(int(math.ceil(kfold * self.train_percent)))]

            for c, i in enumerate(ids):
                if c % kfold in pos:
                    keep_ids = np.append(keep_ids, i[1])

        elif mode == "test":

            start = cv
            start = [(i + start + 1) % kfold for i in range(int(math.ceil(kfold * self.train_percent)))][-1]
            pos = [(i + start) % kfold for i in range(int(math.ceil(kfold * self.test_percent)))]

            for c, i in enumerate(ids):
                if c % kfold in pos:
                    keep_ids = np.append(keep_ids, i[1])

        elif mode == "valid":

            start = cv
            start = [(i + start + 1) % kfold for i in range(int(math.ceil(kfold * (self.train_percent + self.test_percent))))][-1]
            pos = [(i + start) % kfold for i in range(int(math.ceil(kfold * self.valid_percent)))]

            for c, i in enumerate(ids):
                if c % kfold in pos:
                    keep_ids = np.append(keep_ids, i[1])

        return keep_ids, x

    def datafeed(self, mode, config, criteria="pred", min_count=10, shuffle=True, kfold=10, cv=1):
        """

        :param mode: train, valid, test
        :param config: config object
        :param criteria: the column label to do zero shot on "pred" "subtype" "objtype"
        :param shot_percentage: float between 0 and 1 indicating the percentage of the training data taken into consideration
        :param min_count: int indicating the minimum count of the predicates of the examples being taken in to consideration
        :param shuffle: whether to shuffle the training data or not
        :param kfold:
        :param cv:
        :return:
        """

        x = self.read_data()

        dataids, x = self.filter_data(x, mode, criteria, min_count, kfold, cv)
        dataids = [i for i in dataids if i in x.index]

        if shuffle:
            np.random.shuffle(dataids)

        return self.yield_datafeed(mode, dataids, x, config)



