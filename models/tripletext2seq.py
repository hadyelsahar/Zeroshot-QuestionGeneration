from __future__ import print_function
import time
import math
import pickle

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

class TripleText2SeqModel():
    """
    This Model is called triples sequences to sequence model
    model takes a single triple and multiple sequences as an input and outputs
    a single sequence.

    This model is equipped by two attention modules
    - attention over the input triples
    - attention over each encoded vector of each word in the
    input sequences

    - Triple Encoder:
        - Entities Encoded through Entity Embeddings
        - Predicates Encoded Through Predicate Embeddings
    - Sequences Encoder:
        - a separate RNN over Word Embeddings of each input sequence


    Data preparation:
    - Thise model doesn't handle Additional tokens `<unk> <rare> <pad>`
    those are expected to be added beforehand to the vocabulary
    - vocabulary is created offline
    - The inputs to the decoder are preprocessed beforehand to start with  `<s>` and  `<\s>`
    - targets are decoder inputs shifted by one (to ignore start symbol)
    """
    def __init__(self, config, mode='training'):

        print('Initializing new seq 2 seq model')

        assert mode in ['training', 'evaluation', 'inference']

        self.mode = mode
        self.config = config

        self.__create_placeholders()
        self.__create_encoder()
        self.__create_decoder()

    def __create_placeholders(self):
        """
        Function to create placeholders for each
        :return:
        """

        # Encoder Inputs
        #################

        # Input Triple
        ###############

        # The input triple is given in the form of list of entities [sub,obj] and list of predicates [pred]
        # This design allows also inputting multiple triples at once since order matters [s1,s2,o1,o2] [p1,p2]
        self.encoder_entities_inputs = tf.placeholder(tf.int32, shape=[None, self.config.ENTITIESLENGTH], name="encoder_entities_inputs")
        self.encoder_predicates_inputs = tf.placeholder(tf.int32, shape=[None, self.config.PREDICATESLENGTH], name="encoder_predicates_inputs")
        self.encoder_predicates_direction = tf.placeholder(tf.float32, shape=[None], name="encoder_predicates_direction")

        # Input Sequences
        # textual evidences = input sequences
        ######################################

        # input sequences with padding
        # :size =  NUMBER_OF_TEXTUAL_EVIDENCES x BATCHSIZE x input sequence max length
        self.encoder_text_inputs = tf.placeholder(dtype=tf.int32, shape=[self.config.NUMBER_OF_TEXTUAL_EVIDENCES, None, None], name='encoder_text_inputs')
        # actual lengths of each input sequence
        # :size =  NUMBER_OF_TEXTUAL_EVIDENCES x 1
        # each batch has a fixed input sequence length
        self.encoder_text_inputs_length = tf.placeholder(dtype=tf.int32, shape=[self.config.NUMBER_OF_TEXTUAL_EVIDENCES, None], name='encoder_text_inputs_length')

        self.batch_size = tf.shape(self.encoder_entities_inputs)[0]

        # Decoder placeholders:
        # these are the raw inputs to the decoder same as input sequences
        # output sequence with padding
        # :size =  BATCHSIZE x output sequence max length
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
        # number indicating actual lengths of the output sequence
        # :size =  BATCHSIZE x 1
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

        if self.mode == "training":

            self.decoder_inputs_train = self.decoder_inputs

            # for training our targets are decoder inputs shifted by one (to ignore the <s> symbol)
            # as shown in figure https://www.tensorflow.org/images/basic_seq2seq.png
            self.decoder_targets_train = self.decoder_inputs[:, 1:]

            # decoder_inputs_length_train: [batch_size x 1]
            self.decoder_inputs_length_train = self.decoder_inputs_length
            self.decoder_targets_length_train = self.decoder_inputs_length - 1

            # calculating max_decoder_length
            self.decoder_max_length = tf.reduce_max(self.decoder_targets_length_train)

        elif self.mode == "inference":
            # at inference time there's no decoder input so we set the Decode length to a maximum.
            self.decoder_max_length = self.config.MAX_DECODE_LENGTH

        # global step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __build_single_rnn_cell(self, hidden_size):

        cell = tf.nn.rnn_cell.GRUCell(hidden_size)

        # if self.use_dropout:
        #     cell = DropoutWrapper(cell, dtype=self.dtype,
        #                           output_keep_prob=self.keep_prob_placeholder, )

        return cell

    def __create_triple_encoder(self):
        print('building Triples encoder ...')
        start = time.time()

        with tf.variable_scope('encoder'):
            # Create Embeddings Weights

            if self.config.USE_PRETRAINED_KB_EMBEDDINGS:

                ent_kb_emb = pickle.load(open(self.config.PRETRAINED_ENTITIES_EMBEDDINGS_PATH))
                self.encoder_entities_embeddings = tf.Variable(ent_kb_emb, name="entities_embeddings", trainable=self.config.TRAIN_KB_EMBEDDINGS)

                pred_kb_emb = pickle.load(open(self.config.PRETRAINED_PREDICATES_EMBEDDINGS_PATH))
                self.encoder_predicates_embeddings = tf.Variable(pred_kb_emb, name="predicates_embeddings",
                                                               trainable=self.config.TRAIN_KB_EMBEDDINGS)

            else:
                self.encoder_entities_embeddings = tf.get_variable("entities_embeddings",
                                                      shape=[self.config.ENTITIES_VOCAB, self.config.ENTITIES_EMBEDDING_SIZE],
                                                      initializer=self.__helper__initializer(),
                                                      dtype=tf.float32
                                                      )
                self.encoder_predicates_embeddings = tf.get_variable("predicates_embeddings",
                                                              shape=[self.config.PREDICATES_VOCAB,
                                                                     self.config.PREDICATES_EMBEDDING_SIZE],
                                                              initializer=self.__helper__initializer(),
                                                              dtype=tf.float32
                                                      )

            # embedding the encoder inputs
            # encoder_inputs is of size [Batch size x 3]
            # encoder_inputs_embedded is of size [Batch size x 3 x TRIPLES_EMBEDDING_SIZE]
            self.encoder_entities_inputs_embedded = tf.nn.embedding_lookup(self.encoder_entities_embeddings, self.encoder_entities_inputs)

            self.encoder_predicates_inputs_embedded = tf.nn.embedding_lookup(self.encoder_predicates_embeddings, self.encoder_predicates_inputs)

            direction = tf.expand_dims(self.encoder_predicates_direction, axis=1)
            direction = tf.expand_dims(direction, axis=2)

            self.encoder_predicates_inputs_embedded = tf.multiply(self.encoder_predicates_inputs_embedded, direction)

            self.encoder_triples_inputs_embedded = tf.concat((self.encoder_entities_inputs_embedded, self.encoder_predicates_inputs_embedded), axis=1)
            # Encode input triple into a vector
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_triples_last_state = tf.concat(tf.unstack(self.encoder_triples_inputs_embedded, axis=1), axis=1)

        print('Building encoder in: ', time.time() - start, ' secs')

    def __create_seq_encoder(self):

        print('Building Input Sequence Encoder ...')
        start = time.time()

        with tf.variable_scope('encoder'):

            ###################
            # Word Embeddings #
            ###################
            # Create Word Embeddings Weights
            if self.config.USE_PRETRAINED_WORD_EMBEDDINGS:

                word_emb = pickle.load(open(self.config.PRETRAINED_WORD_EMBEDDINGS_PATH)).astype(np.float32)
                self.encoder_word_embeddings = tf.Variable(word_emb, name="encoder_word_embeddings",
                                                           trainable=self.config.TRAIN_WORD_EMBEDDINGS)

            else:
                self.encoder_word_embeddings = tf.get_variable("encoder_word_embeddings",
                                                               shape=[self.config.DECODER_VOCAB_SIZE,
                                                                      self.config.INPUT_SEQ_EMBEDDING_SIZE],
                                                               initializer=self.__helper__initializer(),
                                                               dtype=tf.float32
                                                               )

            # Embedding the encoder inputs
            # Encoder Input size = NUMBER_OF_TEXTUAL_EVIDENCES x BATCH x input_length
            # Embedded Input size =  NUMBER_OF_TEXTUAL_EVIDENCES x BATCH x input_length x word_embeddings_size
            self.encoder_text_inputs_embedded = tf.nn.embedding_lookup(self.encoder_word_embeddings,
                                                                       self.encoder_text_inputs)

            #######
            # RNN #
            #######

            # building a multilayer RNN for each Textual Evidence
            # Encode input sequences into context vectors:
            # encoder_outputs: [Num_text_evidence, batch_size, max_time_step, cell_output_size]
            # encoder_state: [Num_text_evidence, batch_size, cell_output_size]

            self.encoder_text_outputs = []
            self.encoder_text_last_state = []

            # If not bidirectional encoder
            self.encoder_cell = []

            rnn = self.__build_single_rnn_cell(self.config.INPUT_SEQ_RNN_HIDDEN_SIZE)

            if "bi" not in self.config.ENCODER_RNN_CELL_TYPE:
                    for _ in range(self.config.NUMBER_OF_TEXTUAL_EVIDENCES):
                        #rnn = self.__build_single_rnn_cell(self.config.INPUT_SEQ_RNN_HIDDEN_SIZE)
                        self.encoder_cell.append(tf.nn.rnn_cell.MultiRNNCell([rnn] * self.config.NUM_LAYERS))

                    for i in range(self.config.NUMBER_OF_TEXTUAL_EVIDENCES):

                        out, state = tf.nn.dynamic_rnn(
                            cell=self.encoder_cell[i],
                            inputs=self.encoder_text_inputs_embedded[i],
                            sequence_length=self.encoder_text_inputs_length[i],
                            dtype=tf.float32
                        )

                        self.encoder_text_outputs.append(out)
                        self.encoder_text_last_state.append(tf.squeeze(state, axis=0))

            # If bidirectional encoder
            else:
                self.fwd_encoder_cell = []
                self.bw_encoder_cell = []
                for _ in range(self.config.NUMBER_OF_TEXTUAL_EVIDENCES):
                    # two rnn decoders for each layer for each input sequence\
                    #fwrnn = self.__build_single_rnn_cell(self.config.INPUT_SEQ_RNN_HIDDEN_SIZE)
                    #bwrnn = self.__build_single_rnn_cell(self.config.INPUT_SEQ_RNN_HIDDEN_SIZE)

                    self.fwd_encoder_cell.append([rnn] * self.config.NUM_LAYERS)
                    self.bw_encoder_cell.append([rnn] * self.config.NUM_LAYERS)

                for i in range(self.config.NUMBER_OF_TEXTUAL_EVIDENCES):

                    out, fwd_state, bk_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        cells_fw=self.fwd_encoder_cell[i],
                        cells_bw=self.bw_encoder_cell[i],
                        inputs=self.encoder_text_inputs_embedded[i],
                        sequence_length=self.encoder_text_inputs_length[i],
                        dtype=tf.float32
                    )

                    self.encoder_text_outputs.append(tf.concat(out, 2))
                    self.encoder_text_last_state.append(tf.squeeze(tf.concat([fwd_state, bk_state], 2), axis=0))

        print('Building encoder in: ', time.time() - start, ' secs')

    def __create_encoder(self):

        self.__create_triple_encoder()
        self.__create_seq_encoder()

        # concatinating last state of the triple encoder with the last state of each text input being encoded
        last_states = [self.encoder_triples_last_state] + self.encoder_text_last_state

        self.encoder_last_state = tf.concat(last_states, axis=1)

    def __create_decoder_cell(self):

        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.config.DECODER_RNN_HIDDEN_SIZE)

        # fully connected layer to change size of Encoder Last state to Decoder Hidden size
        decoder_hidden_state_reshape = Dense(self.config.DECODER_RNN_HIDDEN_SIZE)

        self.decoder_initial_state = (decoder_hidden_state_reshape(self.encoder_last_state), )


    def __create_decoder_attention_cell_old(self):
        """
        create decoder RNN with attention
        :return:
        """

        memory = tf.concat([self.encoder_triples_inputs_embedded] + self.encoder_text_outputs, axis=1)

        self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.config.TRIPLES_EMBEDDING_SIZE,    # the depth of the Attention layer
            memory=memory,
            name="Attention"
        )

        # create decoder cell:
        gru = self.__build_single_rnn_cell(self.config.DECODER_RNN_HIDDEN_SIZE)
        self.decoder_cell_list = [gru] * self.config.NUM_LAYERS

        decoder_hidden_state_reshape = Dense(self.config.DECODER_RNN_HIDDEN_SIZE)

        self.decoder_cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_layer_size=self.config.DECODER_RNN_HIDDEN_SIZE,     # the output hidden size of the last decoder
            attention_mechanism=self.attention_mechanism,
            initial_cell_state=decoder_hidden_state_reshape(self.encoder_last_state),
            alignment_history=False,
            name="Attention_Wrapper"
        )

        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(self.decoder_cell_list)

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # self.decoder_initial_state = self.encoder_last_state

        init_state = self.decoder_cell_list[-1].zero_state(
            batch_size=self.batch_size,
            dtype=tf.float32
        )

        # a tuple because decode initial state has to take a tuple
        self.decoder_initial_state = (init_state,)


    def __create_decoder_attention_cell(self):
        """
        create decoder RNN with attention
        :return:
        """

        triple_memory = self.encoder_triples_inputs_embedded

        self.triple_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.config.TRIPLES_EMBEDDING_SIZE,    # the depth of the Attention layer
            memory=triple_memory,
            name="TripleAttention"
        )

        context_memory = tf.concat(self.encoder_text_outputs, axis=1)

        self.context_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.config.INPUT_SEQ_RNN_HIDDEN_SIZE if "bi" not in self.config.ENCODER_RNN_CELL_TYPE
            else self.config.INPUT_SEQ_RNN_HIDDEN_SIZE * 2,    # the depth of the Attention layer
            memory=context_memory,
            name="ContextAttention"
        )

        # create decoder cell:
        gru = self.__build_single_rnn_cell(self.config.DECODER_RNN_HIDDEN_SIZE)
        self.decoder_cell_list = [gru] * self.config.NUM_LAYERS

        decoder_hidden_state_reshape = Dense(self.config.DECODER_RNN_HIDDEN_SIZE)

        self.decoder_cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            # the output hidden size of the last decoder
            attention_layer_size=[self.config.TRIPLES_EMBEDDING_SIZE,
                                  self.config.INPUT_SEQ_RNN_HIDDEN_SIZE if "bi" not in self.config.ENCODER_RNN_CELL_TYPE
                                  else self.config.INPUT_SEQ_RNN_HIDDEN_SIZE * 2],
            attention_mechanism=[self.triple_attention_mechanism, self.context_attention_mechanism],
            initial_cell_state=decoder_hidden_state_reshape(self.encoder_last_state),
            alignment_history=False,
            name="Attention_Wrapper"
        )

        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(self.decoder_cell_list)

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # self.decoder_initial_state = self.encoder_last_state

        init_state = self.decoder_cell_list[-1].zero_state(
            batch_size=self.batch_size,
            dtype=tf.float32
        )

        # a tuple because decode initial state has to take a tuple
        self.decoder_initial_state = (init_state,)

    def __create_decoder(self):

        print("building decoder and attention ..")
        start = time.time()

        with tf.variable_scope('decoder'):

            # input and output layers to the decoder
            # decoder_input_layer = Dense(self.config.DECODER_RNN_HIDDEN_SIZE, dtype=tf.float32, name='decoder_input_projection')
            decoder_output_layer = Dense(self.config.DECODER_VOCAB_SIZE, name="decoder_output_projection")

            if self.config.COUPLE_ENCODER_DECODER_WORD_EMBEDDINGS:
                # connect encoder and decoder word embeddings
                self.decoder_embeddings = self.encoder_word_embeddings

            elif self.config.USE_PRETRAINED_WORD_EMBEDDINGS:

                word_emb = pickle.load(open(self.config.PRETRAINED_WORD_EMBEDDINGS_PATH)).astype(np.float32)

                self.decoder_embeddings = tf.Variable(word_emb, name="decoder_embeddings",
                                                               trainable=self.config.TRAIN_WORD_EMBEDDINGS)

            else:
                self.decoder_embeddings = tf.get_variable("decoder_embeddings",
                                                      shape=[self.config.DECODER_VOCAB_SIZE, self.config.DECODER_EMBEDDING_SIZE],
                                                      initializer=self.__helper__initializer(),
                                                      dtype=tf.float32
                                                      )

            if self.config.USE_ATTENTION:
                self.__create_decoder_attention_cell()
            else:
                self.__create_decoder_cell()

            ######################################
            # Build the decoder in training mode #
            ######################################
            if self.mode == 'training':

                # changing inputs to embeddings and then through the input projection
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(params=self.decoder_embeddings,
                                                                      ids=self.decoder_inputs_train)
                # self.decoder_inputs_embedded = decoder_input_layer(self.decoder_inputs_embedded)

                # Helper to feed inputs to the training:

                self.training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_inputs_embedded,
                    sequence_length=self.decoder_inputs_length_train,
                    name='training_helper')

                # Build the decoder
                self.training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=self.training_helper,
                    initial_state=self.decoder_initial_state,
                    output_layer=decoder_output_layer)

                # decoder outputs are of type tf.contrib.seq2seq.BasicDecoderOutput
                # has two fields `rnn_output` and `sample_id`

                self.decoder_outputs_train, self.decoder_last_state_train, self.decoder_outputs_length_decode_train = tf.contrib.seq2seq.dynamic_decode(
                    decoder=self.training_decoder,
                    impute_finished=True,
                    maximum_iterations=self.decoder_max_length
                )

                # In the training mode only create LOSS and Optimizer

                self.__create_loss()
                self.__create_optimizer()

            ######################################
            # Build the decoder in sampling mode #
            ######################################
            elif self.mode == 'inference':

                start_tokens = tf.ones([self.batch_size, ], dtype=tf.int32) * self.config.DECODER_START_TOKEN_ID
                end_token = self.config.DECODER_END_TOKEN_ID

                def decoder_inputs_embedder(inputs):
                    # return decoder_input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))
                    return tf.nn.embedding_lookup(self.decoder_embeddings, inputs)

                # end token is needed so the helper stop feeding new inputs again once the <end> mark is shown.
                decoder_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_inputs_embedder, start_tokens, end_token)

                # Basic decoder performs greedy decoding at each time step
                print("Building Greedy Decoder ...")

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                    helper=decoder_helper,
                                                                    initial_state=self.decoder_initial_state,
                                                                    output_layer=decoder_output_layer)

                self.decoder_outputs_inference, self.decoder_last_state_inference, self.decoder_outputs_length_inference = tf.contrib.seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    maximum_iterations=self.decoder_max_length
                )

                self.decoder_pred_inference = tf.expand_dims(self.decoder_outputs_inference.sample_id, -1)

        print('Building decoder in: ', time.time() - start, ' secs')

    def __create_loss(self):

        print('Creating loss...')
        start = time.time()

        self.decoder_logits = tf.identity(self.decoder_outputs_train.rnn_output, name="decoder_logits")
        self.decoder_pred = tf.argmax(self.decoder_logits, axis=-1, name="decoder_pred")

        # masking the sequence in order to calculate the error according to the calculated
        mask = tf.sequence_mask(self.decoder_inputs_length_train, maxlen=self.decoder_max_length, dtype=tf.float32,
                                name="masks")

        # Control loss dimensions with `average_across_timesteps` and `average_across_batch`
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits,
                                                     targets=self.decoder_targets_train,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False,
                                                     weights=mask,
                                                     name="batch_loss")

        print('Building loss in: ', time.time() - start, ' secs')

    def __create_optimizer(self):
        print('creating optimizer...')
        start = time.time()

        learning_rate = tf.train.exponential_decay(self.config.LR, self.global_step, 200, 0.97, staircase=True)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # normalize the gradients of a parameter vector when its L2 norm exceeds a certain threshold according to
        trainable_params = tf.trainable_variables()

        # calculate gradients of the loss given all the trainable parameters
        gradients = tf.gradients(self.loss, trainable_params)

        # Gradient clipping: new_gradients = gradients * threshold / l2_norm(gradients)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config.MAX_GRAD_NORM)

        self.updates = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

        print('Building optimizer in: ', time.time() - start, ' secs')

    def __helper__initializer(self):
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
        return initializer

    def train(self, sess, encoder_triples_inputs, encoder_text_inputs, encoder_text_inputs_length, decoder_inputs, decoder_inputs_lengths, encoder_predicates_direction):

        feed_dict = {
            # self.encoder_triples_inputs: encoder_triples_inputs,
            self.encoder_entities_inputs: encoder_triples_inputs[:, [0, 2]],  # pick up subjects and objects
            self.encoder_predicates_inputs: encoder_triples_inputs[:, [1]],  # pick up predicates
            self.encoder_text_inputs: encoder_text_inputs,
            self.encoder_text_inputs_length: encoder_text_inputs_length,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_lengths,
            self.encoder_predicates_direction: encoder_predicates_direction
        }
        _, loss = sess.run([self.updates, self.loss], feed_dict=feed_dict)

        return loss

    def eval(self, sess, encoder_triples_inputs, encoder_text_inputs, encoder_text_inputs_length, decoder_inputs, decoder_inputs_lengths, encoder_predicates_direction):
        """
        Run a evaluation step of the model feeding the given inputs
        :param sess:
        :param encoder_inputs:
        :param encoder_inputs_length:
        :param decoder_inputs:
        :param decoder_inputs_lengths:
        :return:
        """

        feed_dict = {
            # self.encoder_triples_inputs: encoder_triples_inputs,
            self.encoder_entities_inputs: encoder_triples_inputs[:, [0, 2]],  # pick up subjects and objects
            self.encoder_predicates_inputs: encoder_triples_inputs[:, [1]],  # pick up predicates
            self.encoder_text_inputs: encoder_text_inputs,
            self.encoder_text_inputs_length: encoder_text_inputs_length,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_lengths,
            self.encoder_predicates_direction: encoder_predicates_direction
        }
        _, loss = sess.run([self.updates, self.loss], feed_dict=feed_dict)

        return loss

    def predict(self, sess, encoder_triples_inputs, encoder_text_inputs, encoder_text_inputs_length, encoder_predicates_direction):
        """
        predict the output given an input
        """

        feed_dict = {
            # self.encoder_triples_inputs: encoder_triples_inputs,
            self.encoder_entities_inputs: encoder_triples_inputs[:, [0, 2]],     # pick up subjects and objects
            self.encoder_predicates_inputs: encoder_triples_inputs[:, [1]],   # pick up predicates
            self.encoder_text_inputs: encoder_text_inputs,
            self.encoder_text_inputs_length: encoder_text_inputs_length,
            self.encoder_predicates_direction: encoder_predicates_direction
        }

        output = sess.run([self.decoder_pred_inference], feed_dict=feed_dict)

        return output[0]

    def save(self, sess, path, var_list=None, global_step=None):

        saver = tf.train.Saver(var_list)
        path = saver.save(sess, save_path=path, global_step=global_step)
        print("model saved in %s" % path)
        return path

    def restore(self, sess, path, var_list=None):
        """
        restore trained model from a specific path
        :param sess:
        :param path:
        :param var_list: if None restore all list
        :return:
        """

        saver = tf.train.Saver(var_list)
        saver.restore(sess, path)
        print("model restored from %s" % path)
