import torch
from torch.nn.functional import cross_entropy

from coda.surrogates import Ensemble, OracleSurrogate


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

LOSS_FNS = {
    # 'ce': cross_entropy, # TODO this won't work out of the box; we don't have logits
    'acc': accuracy_loss
}

SURROGATES = {
    'naive': Ensemble,
    'oracle': OracleSurrogate, 
}
