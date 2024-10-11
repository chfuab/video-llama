from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class VQA_acc:
    def __init__(self, dummy):
        self.dummy = dummy

    def compute_score(pred, gt):
        score = 0
        return score

metrics_mapping = {
    "Bleu": Bleu(4),
    # "METEOR": Meteor(),
    "ROUGE_L": Rouge(),
    "CIDEr": Cider(),
    # "SPICE": Spice(),
}

""" class Caption_scorer():
    def __init__(self):
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
    
    def compute_scores(self, ref, gt):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gt, ref)
            if type(method) == list:
                total_scores["Bleu"] = score
            else:
                total_scores[method] = score
        
        return total_scores """