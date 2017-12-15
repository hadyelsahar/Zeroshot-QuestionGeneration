import os
import pickle

class Config():

    def __init__(self, data, model_name, mode="QG"):

        assert mode in ["QG", "KBEMBED"]

        self.CHECKPOINTS_PATH = os.path.join("./checkpoints/"+model_name, model_name)

        # Triples:
        self.TRIPLELENGTH = 3  # (s,p,o) # make 4 to extend to quads
        self.ENTITIESLENGTH = 2
        self.PREDICATESLENGTH = 1
        self.USE_PRETRAINED_KB_EMBEDDINGS = False

        # knowledge base embeddings training params
        if mode == "KBEMBED":

            self.KB_EMBED_VOCAB_PATH = './data/kb_embeddings_data'

            # counting length of entity and properties vocab

            with open(os.path.join(self.KB_EMBED_VOCAB_PATH, "entity2id.txt")) as f:
                for ec, l in enumerate(f):
                    pass

            with open(os.path.join(self.KB_EMBED_VOCAB_PATH, "relation2id.txt")) as f:
                for pc, l in enumerate(f):
                    pass

            self.ENTITIES_VOCAB = ec - 1    # Size of the encoding vocabulary  # -2  count of first line and remove of header
            self.PREDICATES_VOCAB = pc - 1

            self.PICKLE_ENTITIES_VOCAB = len(data.entityvocab)   # sizes of vocab to pickle (pick the first # and save) (practically the entities and predicates that only appear in simple questions)
            self.PICKLE_PREDICATES_VOCAB = len(data.propertyvocab)

            # KBEmbeddings TransE configurations
            self.L1_flag = True
            self.nbatches = 50
            self.TRANSX_EPOCHS = 10000
            self.margin = 1.0

            self.ENTITIES_EMBEDDING_SIZE = 100
            self.PREDICATES_EMBEDDING_SIZE = 100

            self.LOG_FREQUENCY = 20
            self.SAVE_FREQUENCY = 300

        # MODE QUESTION GENERATION
        else:

            if self.USE_PRETRAINED_KB_EMBEDDINGS:

                self.PRETRAINED_ENTITIES_EMBEDDINGS_PATH = "./checkpoints/transe_old/ent_embeddings.pkl"
                self.PRETRAINED_PREDICATES_EMBEDDINGS_PATH = "./checkpoints/transe_old/rel_embeddings.pkl"
                # infer size from given pickle file
                self.ENTITIES_VOCAB, self.ENTITIES_EMBEDDING_SIZE = pickle.load(open(self.PRETRAINED_ENTITIES_EMBEDDINGS_PATH)).shape
                self.PREDICATES_VOCAB, self.PREDICATES_EMBEDDING_SIZE = pickle.load(open(self.PRETRAINED_PREDICATES_EMBEDDINGS_PATH)).shape
                self.TRAIN_KB_EMBEDDINGS = False     # make preloaded embeddings fixed

                # for now has to be the same size as the triples embedding size
                # because all are being merged into one attention memory
                self.TYPES_RNN_HIDDEN_SIZE = self.ENTITIES_EMBEDDING_SIZE

            else:

                # define them manually
                self.ENTITIES_EMBEDDING_SIZE = 200
                self.PREDICATES_EMBEDDING_SIZE = 200
                self.ENTITIES_VOCAB = len(data.entityvocab)  # Size of the encoding vocabulary
                self.PREDICATES_VOCAB = len(data.propertyvocab)
                self.TRAIN_KB_EMBEDDINGS = True
                # for now has to be the same size as the triples embedding size
                # because all are being merged into one attention memory
                self.TYPES_RNN_HIDDEN_SIZE = 200

            self.TRIPLES_EMBEDDING_SIZE = self.ENTITIES_EMBEDDING_SIZE


            # Types: a.k.a Words
            self.USE_PRETRAINED_WORD_EMBEDDINGS = False
            self.NUMBER_OF_TEXTUAL_EVIDENCES = 3   # Subject type, Object Type, Predicate Type

            if self.USE_PRETRAINED_WORD_EMBEDDINGS:

                self.PRETRAINED_WORD_EMBEDDINGS_PATH = "./data/wordembeddings/glove100d.pkl"
                self.TYPES_ENCODER_VOCAB, self.TYPES_EMBEDDING_SIZE = pickle.load(open(self.PRETRAINED_WORD_EMBEDDINGS_PATH)).shape
                self.TRAIN_WORD_EMBEDDINGS = True

            else:
                self.TYPES_ENCODER_VOCAB = len(data.wordvocab)
                self.TYPES_EMBEDDING_SIZE = 50

            # Decoder:
            self.NUM_LAYERS = 1

            self.COUPLE_ENCODER_DECODER_WORD_EMBEDDINGS = True

            if self.COUPLE_ENCODER_DECODER_WORD_EMBEDDINGS:
                self.DECODER_EMBEDDING_SIZE = self.TYPES_EMBEDDING_SIZE
                self.DECODER_VOCAB_SIZE = self.TYPES_ENCODER_VOCAB
            else:
                self.DECODER_EMBEDDING_SIZE = 50
                self.DECODER_VOCAB_SIZE = len(data.wordvocab)  # Size of the decoding vocabulary


            self.DECODER_RNN_HIDDEN_SIZE = 500

            # Attention:
            self.USE_ATTENTION = True
            self.ATTENTION_HIDDEN_SIZE = self.TRIPLES_EMBEDDING_SIZE

            # Inference
            self.DECODER_START_TOKEN_ID = 2     # numbers from the vocab file
            self.DECODER_END_TOKEN_ID = 3       # numbers from the vocab file
            self.MAX_DECODE_LENGTH = 12         # Arbitary number based on data analysis  # only used in inference time

            # Training Params
            self.BATCH_SIZE = 500
            self.LR = 0.001
            self.MAX_GRAD_NORM = 0.1
            self.MAX_EPOCHS = 250
            self.LOG_FREQUENCY = 20
            self.SAVE_FREQUENCY = 200          # save per global step
