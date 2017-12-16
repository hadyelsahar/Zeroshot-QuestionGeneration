from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider

class Evaluator:
    def __init__(self, predicted_list, label_list):
        """

        :param predicted_list: list of predicted strings
        :param label_list: list of lists of true labels 1 item can have several true labels
        """

        self.eval = []
        self.overall_eval = {}
        self.id_to_eval = {}

        self.predicted_list = [[i] for i in predicted_list]

        if type(label_list[0]) is str:
            label_list = [[i] for i in label_list]
        self.label_list = label_list

    def evaluate(self):

        # imgIds = self.coco.getImgIds()
        gts = dict(zip(range(0, len(self.predicted_list)), self.predicted_list))
        res = dict(zip(range(0, len(self.label_list)), self.label_list))

        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.set_textid_to_eval(scs, gts.keys(), m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.set_textid_to_eval(scores, gts.keys(), method)
                print "%s: %0.3f"%(method, score)
        self.set_eval()

    def setEval(self, score, method):
        self.overall_eval[method] = score

    def set_textid_to_eval(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.id_to_eval:
                self.id_to_eval[imgId] = {}
                self.id_to_eval[imgId]["image_id"] = imgId
            self.id_to_eval[imgId][method] = score

    def set_eval(self):
        self.eval = [eval for imgId, eval in self.id_to_eval.items()]