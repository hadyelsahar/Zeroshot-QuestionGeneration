from __future__ import print_function

import time
import math
import pickle

import tensorflow as tf
from tensorflow.python.layers.core import Dense

class Triple2SeqModel():
    """
    This Model Triple to sequence model:

    Generating Factoid Questions With Recurrent Neural Networks: The 30M Factoid Question-Answer Corpus
    https://arxiv.org/abs/1603.06807

    - Triple Encoder:
        - Entities Encoded through Entity Embeddings
        - Predicates Encoded Through Predicate Embeddings

    Data preparation:
    - This model doesn't handle Additional tokens `<unk> <rare> <pad>` those are expected to be added beforehand to the vocabulary
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
        # self.__create_encoder()
        self.__create_triple_encoder()
        self.__create_decoder()

    def __create_placeholders(self):

        # encoder_inputs : size [batch_size, triples_size(normally 3)]
        self.encoder_entities_inputs = tf.placeholder(tf.int32, shape=[None, self.config.ENTITIESLENGTH], name="encoder_entities_inputs")
        self.encoder_predicates_inputs = tf.placeholder(tf.int32, shape=[None, self.config.PREDICATESLENGTH], name="encoder_predicates_inputs")
        self.encoder_predicates_direction = tf.placeholder(tf.float32, shape=[None], name="encoder_predicates_direction")

        self.batch_size = tf.shape(self.encoder_entities_inputs)[0]

        # Decoder placeholders:
        # these are the raw inputs to the decoder:
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
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

    def __create_triple_encoder(self):
        print('building encoder ...')
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
                                                                   shape=[self.config.ENTITIES_VOCAB,
                                                                          self.config.ENTITIES_EMBEDDING_SIZE],
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
            self.encoder_entities_inputs_embedded = tf.nn.embedding_lookup(self.encoder_entities_embeddings,
                                                                           self.encoder_entities_inputs)
            self.encoder_predicates_inputs_embedded = tf.nn.embedding_lookup(self.encoder_predicates_embeddings,
                                                                             self.encoder_predicates_inputs)

            direction = tf.expand_dims(self.encoder_predicates_direction, axis=1)
            direction = tf.expand_dims(direction, axis=2)

            self.encoder_predicates_inputs_embedded = tf.multiply(self.encoder_predicates_inputs_embedded, direction)

            self.encoder_triples_inputs_embedded = tf.concat(
                (self.encoder_entities_inputs_embedded, self.encoder_predicates_inputs_embedded), axis=1)

            # Encode input triple into a vector
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_triples_last_state = tf.concat(tf.unstack(self.encoder_triples_inputs_embedded, axis=1), axis=1)

        print('Building encoder in: ', time.time() - start, ' secs')

    def __build_single_rnn_cell(self, hidden_size):

        cell = tf.nn.rnn_cell.GRUCell(hidden_size)

        # if self.use_dropout:
        #     cell = DropoutWrapper(cell, dtype=self.dtype,
        #                           output_keep_prob=self.keep_prob_placeholder, )
        # if self.use_residual:
        #     cell = ResidualWrapper(cell)

        return cell

    def __create_decoder_cell(self):

        gru = tf.nn.rnn_cell.GRUCell(self.config.DECODER_RNN_HIDDEN_SIZE)

        self.decoder_cell_list = [gru] * self.config.NUM_LAYERS

        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(self.decoder_cell_list)

        decoder_hidden_state_reshape = Dense(self.config.DECODER_RNN_HIDDEN_SIZE)  # reshape last state of encoder to decoder hidden size
        self.decoder_initial_state = (decoder_hidden_state_reshape(self.encoder_triples_last_state), )

    def __create_decoder_attention_cell(self):
        """
        create decoder RNN with attention
        :return:
        """

        self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.config.TRIPLES_EMBEDDING_SIZE,    # the depth of the Attention layer
            memory=self.encoder_triples_inputs_embedded,
            name="Attention"
        )

        # create decoder cell:
        gru = self.__build_single_rnn_cell(self.config.DECODER_RNN_HIDDEN_SIZE)
        self.decoder_cell_list = [gru] * self.config.NUM_LAYERS

        decoder_hidden_state_reshape = Dense(self.config.DECODER_RNN_HIDDEN_SIZE)  # reshape last state of encoder to decoder hidden size
        self.decoder_cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_layer_size=self.config.DECODER_RNN_HIDDEN_SIZE,         # the output hidden size of the last decoder
            attention_mechanism=self.attention_mechanism,
            initial_cell_state= decoder_hidden_state_reshape(self.encoder_triples_last_state),
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
            decoder_input_layer = Dense(self.config.DECODER_RNN_HIDDEN_SIZE, dtype=tf.float32, name='decoder_input_projection')
            decoder_output_layer = Dense(self.config.DECODER_VOCAB_SIZE, name="decoder_output_projection")

            # creating decoder embedding weights
            self.decoder_embeddings = tf.get_variable("decoder_embeddings",
                                                      shape=[self.config.DECODER_VOCAB_SIZE, self.config.DECODER_RNN_HIDDEN_SIZE],
                                                      initializer=self.__helper__initializer(),
                                                      dtype=tf.float32
                                                      )


            self.__create_decoder_attention_cell()
            # self.__create_decoder_cell()

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
                    return tf.nn.embedding_lookup(self.decoder_embeddings, inputs)

                # end token is needed so the helper stop feeding new inputs again once the <end> mark is shown.
                # expected decoder output will be a set of <end> <end> <end> <end> words until the max sequence length is shown.
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

        # learning_rate = tf.train.exponential_decay(self.config.LR, self.global_step, 100, 0.96, staircase=True)

        # self.opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
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


    def train(self, sess, encoder_inputs, decoder_inputs, decoder_inputs_lengths, encoder_predicates_direction):

        feed_dict = {
            self.encoder_entities_inputs: encoder_inputs[:, [0, 2]],  # pick up subjects and objects
            self.encoder_predicates_inputs: encoder_inputs[:, [1]],  # pick up predicates
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_lengths,
            self.encoder_predicates_direction: encoder_predicates_direction
        }
        _, loss = sess.run([self.updates, self.loss], feed_dict=feed_dict)

        return loss

    def eval(self, sess, encoder_triples_inputs, decoder_inputs, decoder_inputs_lengths, encoder_predicates_direction):
        """
        Run a evaluation step of the model feeding the given inputs
        :param sess:
        :param encoder_inputs:
        :param decoder_inputs:
        :param decoder_inputs_lengths:
        :return:
        """

        feed_dict = {
            self.encoder_entities_inputs: encoder_triples_inputs[:, [0, 2]],  # pick up subjects and objects
            self.encoder_predicates_inputs: encoder_triples_inputs[:, [1]],  # pick up predicates
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_lengths,
            self.encoder_predicates_direction: encoder_predicates_direction
        }
        _, loss = sess.run([self.updates, self.loss], feed_dict=feed_dict)

        return loss

    def predict(self, sess, encoder_triples_inputs, encoder_predicates_direction):
        """
        predict the output given an input
        """

        feed_dict = {
            self.encoder_entities_inputs: encoder_triples_inputs[:, [0, 2]],  # pick up subjects and objects
            self.encoder_predicates_inputs: encoder_triples_inputs[:, [1]],  # pick up predicates
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


#################
# Model Testing #
#################

if __name__ == "__main__":

    import numpy as np
    import os

    class config():

        # Triples:
        TRIPLELENGTH = 3   # (s,p,o) # make 4 to extend to quads
        ENTITIESLENGTH = 2
        PREDICATESLENGTH = 1

        ENTITIES_EMBEDDING_SIZE = 30
        PREDICATES_EMBEDDING_SIZE = 30
        TRIPLES_EMBEDDING_SIZE = ENTITIES_EMBEDDING_SIZE
        ENTITIES_VOCAB = 12  # Size of the encoding vocabulary
        PREDICATES_VOCAB = 12

        USE_PRETRAINED_KB_EMBEDDINGS = True

        if USE_PRETRAINED_KB_EMBEDDINGS:

            PRETRAINED_ENTITIES_EMBEDDINGS_PATH = "./checkpoints/transe/ent_embeddings.pkl"
            PRETRAINED_PREDICATES_EMBEDDINGS_PATH = "./checkpoints/transe/rel_embeddings.pkl"
            # infer size from given pickle file
            _, ENTITIES_EMBEDDING_SIZE = pickle.load(open(PRETRAINED_ENTITIES_EMBEDDINGS_PATH)).shape
            _, PREDICATES_EMBEDDING_SIZE = pickle.load(open(PRETRAINED_PREDICATES_EMBEDDINGS_PATH)).shape
            TRAIN_KB_EMBEDDINGS = False     # make preloaded embeddings fixed

        # Decoder:
        NUM_LAYERS = 1
        # DECODER_RNN_HIDDEN_SIZE = TRIPLES_EMBEDDING_SIZE * TRIPLELENGTH
        DECODER_RNN_HIDDEN_SIZE = 44
        DECODER_VOCAB_SIZE = 15  # Size of the decoding vocabulary

        # Attention:
        ATTENTION_HIDDEN_SIZE = TRIPLES_EMBEDDING_SIZE

        # Training Params
        BATCH_SIZE = 500
        LR = 0.5
        MAX_GRAD_NORM = 5.0
        MAX_EPOCHS = 5
        SAVE_FREQUENCY = 50
        CHECKPOINTS_PATH = "../checkpoints/"

        # Inference
        DECODER_START_TOKEN_ID = 1
        DECODER_END_TOKEN_ID = 11
        MAX_DECODE_LENGTH = 4

    def datafeeder(sample_size=50000, epochs=config.MAX_EPOCHS):

        # Triples

        encoder_triples_inputs = np.zeros((sample_size, config.TRIPLELENGTH))

        for c, _ in enumerate(range(0, sample_size)):

            t1 = np.random.randint(1, config.ENTITIES_VOCAB - 1, 1)
            t2 = np.random.randint(1, config.PREDICATES_VOCAB - 1, 1)
            t3 = np.random.randint(1, config.ENTITIES_VOCAB - 1, 1)
            encoder_triples_inputs[c] = np.concatenate((t1, t2, t3))

        encoder_predicates_direction = np.random.choice([-1, 1], (sample_size,))

        # Decoder
        decoder_lengths = np.array([5, 5, 5, 5, 5] * (sample_size / 5)) # create variable lengths
        decoder_inputs = np.zeros([sample_size, 5])

        for c, i in enumerate(decoder_lengths):

            decoder_inputs[c, :i] = np.concatenate(([config.DECODER_START_TOKEN_ID], encoder_triples_inputs[c], [config.DECODER_END_TOKEN_ID]))

        for i in xrange(epochs):
            for j in range(0, sample_size/config.BATCH_SIZE):
                start = j * config.BATCH_SIZE
                end = start + config.BATCH_SIZE
                yield (
                    encoder_triples_inputs[start:end],
                    decoder_inputs[start:end], decoder_lengths[start:end],
                    encoder_predicates_direction[start:end]
                )


    model = Triple2SeqModel(config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for encoder_inputs, decoder_inputs, decoder_lengths, encoder_predicates_direction in datafeeder():

            loss = model.train(sess, encoder_inputs, decoder_inputs, decoder_lengths, encoder_predicates_direction)
            print(loss, model.global_step.eval())

            # Save the model checkpoint
            if model.global_step.eval() % config.SAVE_FREQUENCY == 0:
                print('Saving the model..')
                checkpoint_path = os.path.join(config.CHECKPOINTS_PATH, "seq2seq_model")
                path = model.save(sess, checkpoint_path, global_step=model.global_step)

    #############
    # Inference #
    #############

    # loading back the trained model
    tf.reset_default_graph()
    model = Triple2SeqModel(config, 'inference')

    predicted_ids = []

    with tf.Session() as sess:

        if tf.train.checkpoint_exists(tf.train.latest_checkpoint('../checkpoints/')):
            print('reloading the trained model')

            model.restore(sess=sess, path=tf.train.latest_checkpoint('../checkpoints/'))

            encoder_lengths = np.array([3] * 20)
            encoder_triples_inputs = np.zeros((len(encoder_lengths), 3))

            for c, i in enumerate(encoder_lengths):
                encoder_triples_inputs[c] = np.random.randint(2, 10, 3)

            encoder_predicates_direction = np.random.choice([-1, 1], (20,))

            predicted_ids = model.predict(sess, encoder_triples_inputs=encoder_triples_inputs, encoder_predicates_direction=encoder_predicates_direction)

            for c, i in enumerate(encoder_triples_inputs):
                print(encoder_triples_inputs[c])
                print(predicted_ids[c])
                print("____________________")
