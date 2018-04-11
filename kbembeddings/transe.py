#coding:utf-8

import tensorflow as tf
import os
import ctypes
import pickle


class TransEModel(object):

    def __init__(self, config):

        self.config = config

        entity_total = config.ENTITIES_VOCAB
        size = config.ENTITIES_EMBEDDING_SIZE

        relation_total = config.PREDICATES_VOCAB

        margin = config.margin

        self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[entity_total, size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[relation_total, size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))


    def save(self, sess, var_list=None, global_step=None):

        saver = tf.train.Saver(var_list)
        path = saver.save(sess, save_path=self.config.CHECKPOINTS_PATH, global_step=global_step)
        print("model saved in %s" % path)
        return path

    def restore(self, sess, path=None, var_list=None):
        """
        restore trained model from a specific path
        :param sess:
        :param path:
        :param var_list: if None restore all list
        :return:
        """

        saver = tf.train.Saver(var_list)
        if path is None:
            path = tf.train.latest_checkpoint(os.path.dirname(self.config.CHECKPOINTS_PATH))
        saver.restore(sess, path)
        print("model restored from %s" % path)

    def pickle_embeddings(self, sess, path=None, ent_file_name="ent_embeddings.pkl", rel_file_name="rel_embeddings.pkl", top_entities=None, top_predicates=None):
        """

        :param sess: current tf session
        :param path: path of the folder to save embeddings files in
                     embeddings files are saved under the names
        :param all: A
        :return:
        """

        if path is None:
            path = os.path.dirname(self.config.CHECKPOINTS_PATH)

        entpath = os.path.join(path, ent_file_name)
        relpath = os.path.join(path, rel_file_name)
        # save generated entity embeddings and relation embeddings into a pickle file

        print("dumping embeddings into pickle files in %s" % path)

        ent_pickle = self.ent_embeddings.eval(sess) if top_entities is None else self.ent_embeddings.eval(sess)[:top_entities]
        pickle.dump(ent_pickle, open(entpath, "w"))
        rel_pickle = self.rel_embeddings.eval(sess) if top_predicates is None else self.rel_embeddings.eval(sess)[:top_predicates]
        pickle.dump(rel_pickle, open(relpath, "w"))









