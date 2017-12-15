
import argparse
import tensorflow as tf

from config import Config
from data.data import *
from models.tripletext2seq import TripleText2SeqModel
from models.triples2seq import Triple2SeqModel
from evaluator.eval import Evaluator

parser = argparse.ArgumentParser(description='running experiments of question generation')

parser.add_argument('-setup', '--setup', help='zeroshot, fewshot if not filled = Normal', required=False)
parser.add_argument('-criteria', '--criteria', help='the criteria to use for zero shot "pred" "subtype" "objtype" ', required=False)
parser.add_argument('-datapath', '--datapath', help='path to the preprocessed data folder', required=True)

# adding textual evidence
parser.add_argument('-subtype', '--subtype', action='store_true')
parser.add_argument('-objtype', '--objtype', action='store_true')
parser.add_argument('-pred', '--pred', action='store_true')

parser.add_argument('-min', '--mincount', help='int indicating the minimum count of the predicates of the examples being taken in to consideration', required=False)
parser.add_argument('-kfolds', '--kfolds', help='number of kfolds', required=False)
parser.add_argument('-fold', '--fold', help='fold number', required=False)
parser.add_argument('-mn', '--modelname', help='model name', required=False)
parser.add_argument('-epochs', '--epochs', help='epochs', required=False)
parser.add_argument('-remove_unk', '--remove_unk', action='store_true')  # if mentioned then include unknown textual evidences

parser.add_argument('-o', '--out', help='output file name', required=False)
parser.add_argument('-log', '--logfile', help='file to log outputs of experiment with params', required=False)




args = parser.parse_args()

# determine the experiment type
allowed_exp_types = ["zeroshot", "normal", None]
exp_type = None
assert args.setup in allowed_exp_types

if args.setup is None:
    exp_type = "normal"
else:
    exp_type = args.setup

if exp_type == "zeroshot":
    assert args.criteria is not None

isbaseline = False
if not args.subtype and not args.objtype and not args.pred:
    isbaseline = True

############################
# loading args into config #
############################

text_n = 0
######################
# creating model name#
######################
if args.modelname is None:
    model_name = "triples"
    if args.subtype:
        model_name += "_subtype"
        text_n +=1
    if args.objtype:
        model_name += "_objtype"
        text_n += 1
    if args.pred:
        model_name += "_pred"
        text_n += 1

    if exp_type == "zeroshot":
        model_name = exp_type.upper() + "_%s_" % args.criteria + model_name

    model_name += "_%s" % os.path.normpath(args.datapath).split("/")[-1]

else:
    model_name = args.modelname

if args.kfolds is None:
    kfolds = 10
else:
    kfolds = int(args.kfolds)

if args.fold is None:
    fold = 0
else:
    fold = int(args.fold)


###########################
# Loading Data And Config #
###########################

if exp_type == "normal":
    data = Data(datapath=args.datapath)
    fewshots = 0
    config = Config(data, model_name)

    if args.epochs is not None:
        config.MAX_EPOCHS = int(args.epochs)

    config.NUMBER_OF_TEXTUAL_EVIDENCES = text_n

    traindatafeed = data.datafeed("train", config)
    testdatafeed = data.datafeed("test", config)

elif exp_type == "zeroshot":
    data = ZeroShotsDataFeeder(datapath=args.datapath, remove_unk=args.remove_unk)
    config = Config(data, model_name)

    if args.epochs is not None:
        config.MAX_EPOCHS = int(args.epochs)

    config.NUMBER_OF_TEXTUAL_EVIDENCES = text_n

    traindatafeed = data.datafeed("train", config, args.criteria, kfold=kfolds, cv=fold, min_count=int(args.mincount))
    testdatafeed = data.datafeed("test", config, args.criteria, kfold=kfolds, cv=fold, min_count=int(args.mincount))


if args.out is None:
    args.out = "./results/" + model_name + "_%s_Epochs" % config.MAX_EPOCHS + "_%s_cv" % fold + ".csv"

if args.logfile is None:
    args.logfile = "./log/" + model_name + "_%s_Epochs" % config.MAX_EPOCHS + "_%s_cv" % fold + ".log"

###################
#  START TRAINING #
###################

if isbaseline:
    model = Triple2SeqModel(config)
else:
    model = TripleText2SeqModel(config)

print "Start Training model %s " % model_name
print "\n-----------------------\n"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    iterator = traindatafeed

    for d in iterator:

        encoder_triples_inputs, encoder_subtypes_inputs, encoder_subtypes_inputs_length, encoder_objtypes_inputs, encoder_objtypes_inputs_length, encoder_dep_inputs, encoder_dep_length, decoder_inputs, decoder_inputs_lengths, direction, meta = d

        if isbaseline:

            loss = model.train(sess, encoder_triples_inputs, decoder_inputs, decoder_inputs_lengths, direction)
        else:

            encoder_text_inputs = []
            encoder_text_inputs_length = []
            if args.subtype:
                encoder_text_inputs.append(encoder_subtypes_inputs)
                encoder_text_inputs_length.append(encoder_subtypes_inputs_length)
            if args.objtype:
                encoder_text_inputs.append(encoder_objtypes_inputs)
                encoder_text_inputs_length.append(encoder_objtypes_inputs_length)
            if args.pred:
                encoder_text_inputs.append(encoder_dep_inputs)
                encoder_text_inputs_length.append(encoder_dep_length)

            loss = model.train(sess, encoder_triples_inputs, encoder_text_inputs, encoder_text_inputs_length, decoder_inputs, decoder_inputs_lengths, direction)

        if model.global_step.eval() % config.LOG_FREQUENCY == 0:
            print("Global Step %s Epoch %s  Batch %s \t Loss = %s" % (model.global_step.eval(), meta["epoch"], meta["batch_id"], np.mean(loss)))

        # Save the model checkpoint
        if model.global_step.eval() % config.SAVE_FREQUENCY == 0:

            print('Saving the model..')
            checkpoint_path = os.path.join(config.CHECKPOINTS_PATH)
            path = model.save(sess, checkpoint_path, global_step=model.global_step)

###########
# testing #
###########

tf.reset_default_graph()
results = []

if isbaseline:
    model = Triple2SeqModel(config, 'inference')
else:
    model = TripleText2SeqModel(config, 'inference')

predicted_ids = []


def post_process(s, d):

    d = sorted(d, key=lambda i: len(i[1]))

    for v, k in d:
        s = s.replace(k, v)

    return s

with tf.Session() as sess:

    if tf.train.checkpoint_exists(tf.train.latest_checkpoint(os.path.dirname(config.CHECKPOINTS_PATH))):
        print('reloading the trained model')

        model.restore(sess=sess, path=tf.train.latest_checkpoint(os.path.dirname(config.CHECKPOINTS_PATH)))

        predicted = []
        labels = []

        iterator = testdatafeed

        for d in iterator:

            encoder_triples_inputs, encoder_subtypes_inputs, encoder_subtypes_inputs_length, encoder_objtypes_inputs, encoder_objtypes_inputs_length, encoder_dep_inputs, encoder_dep_length, decoder_inputs, decoder_inputs_lengths, direction, meta = d

            if isbaseline:
                predicted_ids = model.predict(sess, encoder_triples_inputs=encoder_triples_inputs,
                                              encoder_predicates_direction=direction)
            else:

                encoder_text_inputs = []
                encoder_text_inputs_length = []
                if args.subtype:
                    encoder_text_inputs.append(encoder_subtypes_inputs)
                    encoder_text_inputs_length.append(encoder_subtypes_inputs_length)
                if args.objtype:
                    encoder_text_inputs.append(encoder_objtypes_inputs)
                    encoder_text_inputs_length.append(encoder_objtypes_inputs_length)
                if args.pred:
                    encoder_text_inputs.append(encoder_dep_inputs)
                    encoder_text_inputs_length.append(encoder_dep_length)

                predicted_ids = model.predict(sess, encoder_triples_inputs=encoder_triples_inputs,
                                              encoder_text_inputs=encoder_text_inputs,
                                              encoder_text_inputs_length=encoder_text_inputs_length,
                                              encoder_predicates_direction=direction)

            for c, i in enumerate(encoder_triples_inputs):

                sub = data.inv_entityvocab[i[0]]
                pred = data.inv_propertyvocab[i[1]]
                obj = data.inv_entityvocab[i[2]]

                y = " ".join([data.inv_wordvocab[i] for i in np.squeeze(predicted_ids[c], axis=1)]).decode("utf-8")
                y_label = " ".join([data.inv_wordvocab[i] for i in decoder_inputs[c]]).decode("utf-8")

                y_post_proc = post_process(y, meta["placeholder_dict"][c])
                y_label_post_proc = post_process(y_label, meta["placeholder_dict"][c])

                subtype = " ".join([data.inv_wordvocab[i] for i in encoder_subtypes_inputs[c]])
                objtype = " ".join([data.inv_wordvocab[i] for i in encoder_objtypes_inputs[c]])
                dep = " ".join([data.inv_wordvocab[i] for i in encoder_dep_inputs[c]])

                results.append([sub, pred, obj, subtype, objtype, dep, y, y_label, y_post_proc, y_label_post_proc])

                predicted.append(y)
                labels.append(y_label)

        results_df = pd.DataFrame(results, columns=["sub", "pred", "obj", "subtype", "objtype", "dep", "y", "y_label", "y_post_proc", "y_label_post_proc"])
        results_df.to_csv(args.out, encoding="utf-8")

        # remove start
        predicted = [" ".join(i.split()[:-1]).encode('ascii', 'ignore') for i in predicted]
        labels = [" ".join(i.split()[1:-1]).encode('ascii', 'ignore') for i in labels]
        e = Evaluator(predicted, labels)
        e.evaluate()

        s = list()
        s.append(("model_Name", model_name))
        s.append(("DECODER_RNN_HIDDEN_SIZE", config.DECODER_RNN_HIDDEN_SIZE))

        s.append(("COUPLE_ENCODER_DECODER_WORD_EMBEDDINGS", config.COUPLE_ENCODER_DECODER_WORD_EMBEDDINGS))
        if config.USE_PRETRAINED_WORD_EMBEDDINGS:
            s.append(("USE_PRETRAINED_WORD_EMBEDDINGS", config.PRETRAINED_WORD_EMBEDDINGS_PATH))
            s.append(("TRAIN_WORD_EMBEDDINGS", config.TRAIN_WORD_EMBEDDINGS))
        else:
            s.append(("USE_PRETRAINED_WORD_EMBEDDINGS", "FALSE"))

        s.append(("USE_PRETRAINED_KB_EMBEDDINGS", config.USE_PRETRAINED_KB_EMBEDDINGS))
        s.append(("TRAIN_KB_EMBEDDINGS", config.TRAIN_KB_EMBEDDINGS))

        s.append(("Batch Size", config.BATCH_SIZE))
        s.append(("Epochs", config.MAX_EPOCHS))
        s.append(("LR", config.LR))
        s.append(("MAX_GRAD_NORM", config.MAX_GRAD_NORM))

        for i in e.overall_eval.items():
            s.append(i)

        with open(args.logfile, 'a+') as f:
            f.write("\n------------\n")
            for k in s:
                print "%s \t %s" % (k[0], k[1])
                f.write("%s \t %s\n" % (k[0], k[1]))
