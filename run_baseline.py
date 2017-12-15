######################################
# RUN FILES FOR EXPERIMENTS BASELINE #
######################################
import argparse

from config import Config
from data.data import *
from baselines.baselines import IR, RTransE, SELECT

parser = argparse.ArgumentParser(description='running experiments of question generation')
parser.add_argument('-o', '--out', help='output file name', required=False)
parser.add_argument('-baseline', '--baseline', help='baseline name ir, ir_transe, ..', required=True)
parser.add_argument('-log', '--logfile', help='file to log outputs of experiment with params', required=False)
parser.add_argument('-datapath', '--datapath', help='path to the preprocessed data folder', required=True)
# adding textual evidence
parser.add_argument('-min', '--mincount', help='int indicating the minimum count of the predicates of the examples being taken in to consideration', required=False)
parser.add_argument('-kfolds', '--kfolds', help='number of kfolds', required=False)
parser.add_argument('-fold', '--fold', help='fold number', required=False)
parser.add_argument('-mn', '--modelname', help='model name', required=False)
parser.add_argument('-remove_unk', '--remove_unk', action='store_true')
parser.add_argument('-setup', '--setup', help='zeroshot, normal if not filled = Normal', required=False)
parser.add_argument('-criteria', '--criteria', help='the criteria to use for zero shot "pred" "subtype" "objtype" ', required=False)

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

######################
# creating model name#
######################
if args.modelname is None:
    model_name = args.baseline

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
    config.BATCH_SIZE = 10000000   # collect all data in one batch

    traindatafeed = data.datafeed("train", config)
    testdatafeed = data.datafeed("test", config)

elif exp_type == "zeroshot":
    data = ZeroShotsDataFeeder(datapath=args.datapath, remove_unk=args.remove_unk)
    config = Config(data, model_name)
    config.BATCH_SIZE = 10000000  # collect all data in one batch

    traindatafeed = data.datafeed("train", config, args.criteria, kfold=kfolds, cv=fold, min_count=int(args.mincount))
    testdatafeed = data.datafeed("test", config, args.criteria, kfold=kfolds, cv=fold, min_count=int(args.mincount))

if args.out is None:
    args.out = "./results_baselines/" + model_name + "_%s_Epochs" % config.MAX_EPOCHS + "_%s_cv" % fold + ".csv"

_K_ = 5
_RADIUS_ = 0.4
_N_COMPONENTS_ = 200

if args.baseline.lower() == "rtranse":
    baseline = RTransE(config.PRETRAINED_ENTITIES_EMBEDDINGS_PATH, config.PRETRAINED_PREDICATES_EMBEDDINGS_PATH)
    print "loading baseline R_TransE"

elif args.baseline.lower() == "ir_text":
    baseline = IR(TEXT=True)
    print "loading baseline IR_TEXT"

elif "select" in args.baseline.lower():
    baseline = SELECT()
    print "loading baseline SELECT"

##############
# TRAINNINIG #
##############
print "training ..."
d = traindatafeed.next()
encoder_triples_inputs, encoder_subtypes_inputs, encoder_subtypes_inputs_length, encoder_objtypes_inputs, encoder_objtypes_inputs_length, encoder_dep_inputs, encoder_dep_length, decoder_inputs, decoder_inputs_lengths, direction, meta = d
print encoder_triples_inputs.shape
textual_evidence = np.concatenate((encoder_dep_inputs, encoder_objtypes_inputs, encoder_subtypes_inputs), axis=1)
baseline.train(encoder_triples_inputs, decoder_inputs, textual_evidence=textual_evidence)

print "Done Training !!"

###########
# Testing #
###########

results = []

predicted = []
labels = []


def post_process(s, d):

    s = s.split("_END_")[0]
    s += "_END_"

    d = sorted(d, key=lambda i: len(i[1]))

    for v, k in d:
        s = s.replace(k, v)

    return s

for tid, d in enumerate(testdatafeed):

    if tid % 100 == 0:
        print "testing item %s" % tid

    encoder_triples_inputs, encoder_subtypes_inputs, encoder_subtypes_inputs_length, encoder_objtypes_inputs, encoder_objtypes_inputs_length, encoder_dep_inputs, encoder_dep_length, decoder_inputs, decoder_inputs_lengths, direction, meta = d
    textual_evidence = np.concatenate((encoder_dep_inputs, encoder_objtypes_inputs, encoder_subtypes_inputs), axis=1)
    predicted_ids = baseline.test(encoder_triples_inputs, textual_evidence=textual_evidence)

    for c, i in enumerate(encoder_triples_inputs):
        sub = data.inv_entityvocab[i[0]]
        pred = data.inv_propertyvocab[i[1]]
        obj = data.inv_entityvocab[i[2]]

        y = " ".join([data.inv_wordvocab[i] for i in predicted_ids[c]]).decode("utf-8")
        y_label = " ".join([data.inv_wordvocab[i] for i in decoder_inputs[c]]).decode("utf-8")

        y_post_proc = post_process(y, meta["placeholder_dict"][c])
        y_label_post_proc = post_process(y_label, meta["placeholder_dict"][c])

        subtype = " ".join([data.inv_wordvocab[i] for i in encoder_subtypes_inputs[c]])
        objtype = " ".join([data.inv_wordvocab[i] for i in encoder_objtypes_inputs[c]])
        dep = " ".join([data.inv_wordvocab[i] for i in encoder_dep_inputs[c]])

        results.append([sub, pred, obj, subtype, objtype, dep, y, y_label, y_post_proc, y_label_post_proc])

        predicted.append(y)
        labels.append(y_label)

results_df = pd.DataFrame(results,
                          columns=["sub", "pred", "obj", "subtype", "objtype", "dep", "y", "y_label", "y_post_proc",
                                   "y_label_post_proc"])
results_df.to_csv(args.out, encoding="utf-8")

