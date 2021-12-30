import torch
from torch import Tensor
from torch import device as Device
from ignite.metrics import Metric, MetricsLambda
from ignite.metrics.metric import reinit__is_reduced
from typing import Sequence


import sys
sys.path.insert(0, '../../..')


class CustomLoss:
    def __init__(self):
        pass
    def __call__(self, pred, batch):
        pass


class WeightedBCEDecay(CustomLoss):
    def __init__(self, pos_weight, reduction='none', decay_factor=0.98):
        super(WeightedBCEDecay, self).__init__()
        self.pos_weight = torch.tensor(pos_weight)
        self.reduction = reduction
        self.decay_factor = decay_factor

    def __call__(self, preds, labels, epoch):
        weight = self.pos_weight * (self.decay_factor ** epoch)
        loss = torch.nn.BCEWithLogitsLoss(reduction=self.reduction,
                                          pos_weight=weight)
        return loss(preds, labels)


class BinaryThresholdConfusionMatrix(Metric):

    '''
    Implements a confusion matrix over a set of thresholds (inclusive) with GPU support
    Leverage redundant computations for monotonically increasing thresholds
    Accumulate confusion in running count tensor for all thresholds (T,4)
    Supports MetricLambdas to reuse confusion matrix statistics across thresholds
    - Accuracy        ~ returns Tensor with dim (T,)
    - Precision       ~ returns Tensor with dim (T,)
    - Recall          ~ returns Tensor with dim (T,)
    - PrecisionRecall ~ returns (Precision, Recall)
    - F1              ~ returns Tensor with dim (T,)
    
    NOTE:
        Metric should not be directly attached to an engine - please attach lambdas instead as needed
        Assumes all tensors have been detached from computational graph - see .detach()
        some elements of the confusion matrix may be zero and may exhibit
        unexpected behavior. Please monitor experiments as needed
    '''
    
    def __init__(self, thresholds: Tensor, pred_key, label_key, device: Device=torch.device("cpu")):
        self.thresholds = thresholds.to(device)
        self.confusion_matrix = torch.zeros(4, len(thresholds), dtype=torch.long, device=device)
        self.device = device
        # super's init should come last since Metric overrides __getattr__ and that messes with self.<foo>'s behavior
        super().__init__(output_transform=lambda x: x, device=device)
        self.required_output_keys = (pred_key, label_key)
        self._required_output_keys = self.required_output_keys

    @reinit__is_reduced
    def reset(self):
        self.confusion_matrix.fill_(0)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        preds, labels = output                                                            # unpack 
        preds, labels = preds.detach(), labels.detach()                                   # get off graph
        if preds.device != self.device: preds = preds.to(self.device)                     # coord device
        if labels.device != self.device: labels = labels.to(self.device)                  # coord device
        preds, labels = preds.view(-1), labels.view(-1)                                   # flatten
        preds, locs = torch.sort(preds)                                                   # sort
        labels = torch.cumsum(labels[locs], 0)                                            # pool for reuse
        labels = torch.cat([torch.tensor([0], device=self.device), labels], dim=0)        # pre-pending 0
        changes = torch.searchsorted(preds, self.thresholds, right=True)                  # get threshold change index
        neg_preds = labels[changes]                                                       # get fwd change accumulation
        pos_preds = labels[-1] - neg_preds                                                # get bck change accumulation
        self.confusion_matrix[0] += (pos_preds).type(torch.long)                          # TP 
        self.confusion_matrix[1] += (len(labels) - 1 - changes - pos_preds).type(torch.long)# FP (-1 accounts for prepend. 0)
        self.confusion_matrix[2] += (changes - neg_preds).type(torch.long)                # TN
        self.confusion_matrix[3] += (neg_preds).type(torch.long)                          # FN

    def compute(self):
        return self.confusion_matrix

def btcmPrecision(cm: BinaryThresholdConfusionMatrix, reduce=True) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    precision = cm[0] / (cm[0]+cm[1] + 1e-15)
    return precision.max() if reduce else precision

def btcmRecall(cm: BinaryThresholdConfusionMatrix, reduce=True) -> MetricsLambda:
    cm = cm.type(torch.DoubleTensor)
    recall = cm[0] / (cm[0]+cm[3] + 1e-15) 
    return recall.max() if reduce else recall

def btcmPrecisionRecall(cm: BinaryThresholdConfusionMatrix) -> MetricsLambda:
    precision_recall = (btcmPrecision(cm, False), btcmRecall(cm, False))
    return precision_recall

def btcmF1(cm: BinaryThresholdConfusionMatrix, reduce=True) -> MetricsLambda:
    precision, recall = btcmPrecisionRecall(cm)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    return f1.max() if reduce else f1

class ApproximateMetrics():
    '''
        This class can be used to approximate ROC type statistics via preset thresholds
        It should be known that this method may under/over estimate true AUCs depending on T
        - ApproxPR_AUC      ~ returns float of Precision-Recall AUC over T
        - ApproxROC_AUC     ~ returns float of ROC AUC over T (worst case, under-estimated)
    '''
    
    @staticmethod
    def ApproxPR_AUC(cm:BinaryThresholdConfusionMatrix) -> MetricsLambda:
        precision, recall = btcmPrecisionRecall(cm)
        auc = -1 * MetricsLambda(torch.trapz, precision, recall)
        return auc
    
    @staticmethod
    def ApproxROC_AUC(cm:BinaryThresholdConfusionMatrix) -> MetricsLambda:
        tpr = btcmRecall(cm, False)
        fpr = cm[1] / (cm[1] + cm[2] + 1e-15)
        auc = -1 * MetricsLambda(torch.trapz, tpr, fpr)
        return auc
