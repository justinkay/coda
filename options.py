import torch
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

import datasets
from acquisition import iid_acquisition, lure_acquisition_ce, lure_acquisition_acc, ams_acquisition_ce_regret, ams_acquisition_acc_regret
from estimators import EmpiricalRisk, LUREEstimator, ASIEstimator, ASEEstimator, ASEPPIEstimator
from surrogates import Ensemble, WeightedEnsemble, TrainableEnsemble, OracleSurrogate, EMsemble
from posteriors import eig_q
from oracle import Oracle, WILDSOracle, MODELSELECTOROracle

def accuracy_loss(preds, labels, **kwargs):
    """Get 1 - accuracy (a loss), nonreduced. Handles whether we are working with scores or integer labels."""
    if len(labels.shape) > 1:
        argmaxed_preds = torch.argmax(preds, dim=-1)
        argmaxed_labels = torch.argmax(labels, dim=-1)
        accs = (argmaxed_preds == argmaxed_labels).float()
    else:
        argmaxed = torch.argmax(preds, dim=-1)
        accs = (argmaxed == labels).float()

    # make it a loss
    return 1 - accs


DATASETS = {
    'domainnet126': datasets.DomainNet126,
    'wilds': datasets.WILDSDataset,
    'modelselector': datasets.MODELSELECTORDataset
}

LOSS_FNS = {
    'ce': cross_entropy,
    'acc': accuracy_loss
}

ACCURACY_FNS = {
    'domainnet126': {
        'acc': Accuracy(task="multiclass", num_classes=126, average="micro"),
        'macro': Accuracy(task="multiclass", num_classes=126, average="macro"), # Musgrave et al use macro average
    },
    'wilds': { # TODO!
        'acc': Accuracy(task="multiclass", num_classes=126, average="micro"),
        'macro': Accuracy(task="multiclass", num_classes=126, average="macro"), # Musgrave et al use macro average
    },
    'modelselector': { # TODO!
        'acc': Accuracy(task="multiclass", num_classes=126, average="micro"),
        'macro': Accuracy(task="multiclass", num_classes=126, average="macro"), # Musgrave et al use macro average
    }
}

SURROGATES = {
    'naive': Ensemble,
    'weighted': WeightedEnsemble,
    'trainable': TrainableEnsemble,
    'oracle': OracleSurrogate, 
    'emsemble': EMsemble
}

ESTIMATORS = {
    'empirical': EmpiricalRisk,
    'lure': LUREEstimator,
    'asi': ASIEstimator,
    'ase': ASEEstimator,
    'ase-ppi': ASEPPIEstimator
}

Q_FNS = {
    'iid': iid_acquisition,
    'lure_ce': lure_acquisition_ce,
    'lure_acc': lure_acquisition_acc,
    'regret_ce': ams_acquisition_ce_regret,
    'regret_acc': ams_acquisition_acc_regret,
    'eig': eig_q
}

ORACLES = {
    'domainnet126': Oracle,
    'wilds': WILDSOracle, 
    'modelselector': MODELSELECTOROracle
}