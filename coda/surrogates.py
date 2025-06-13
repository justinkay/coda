import torch
import torch.nn.functional as F


def losses(surrogate, preds, loss_fn):
    """
    Get losses of some other predictions (pred_logits) compared to this surrogate.
    Could probably remove the batching.
    """
    H, N, C = preds.shape
    surrogate_probs = surrogate.get_preds() # Shape: (N, C)
    surrogate_labels = torch.argmax(surrogate_probs, dim=-1)  # Shape: (N,)

    batch_size = 1000
    losses = torch.zeros(H, N, device=preds.device)
    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        batch_preds = preds[:, i:batch_end, :]  # Shape: (H, batch_size, C)
        batch_preds = batch_preds.reshape(-1, C)  # Shape: (H * batch_size, C)
        batch_surrogate_labels = surrogate_labels[i:batch_end].repeat(H)  # Shape: (H * batch_size,)

        batch_loss = loss_fn(
            batch_preds,
            batch_surrogate_labels.long()
        ).view(H, -1)

        losses[:, i:batch_end] = batch_loss

    return losses

def expected_error(surrogate, preds):
    """
    Get expected error (``accuracy losses'') of some other predictions (pred_logits) compared to this surrogate.
    """
    H, N, C = preds.shape
    pi_y = surrogate.get_preds() # Shape: (N, C)

    pred_classes = preds.argmax(dim=-1)
    y_star_probs = pi_y[torch.arange(pi_y.shape[0]), pred_classes]
    exp_accuracy_loss = 1 - y_star_probs

    return exp_accuracy_loss


class Ensemble:
    def __init__(self, preds, **kwargs):
        self.preds = preds
        self.device = preds.device
        H, N, C = preds.shape

    def get_preds(self, **kwargs):
        return self.preds.mean(dim=0)


class WeightedEnsemble(Ensemble):
    def __init__(self, preds, **kwargs):
        super().__init__(preds)

    def get_preds(self, weights):
        return (self.preds * weights.view(-1, 1, 1)).sum(dim=0)


class OracleSurrogate:
    def __init__(self, oracle, **kwargs):
        self.oracle = oracle
        self.device = oracle.device
        self.preds = self.oracle.dataset.preds # this is dumb

    def get_preds(self, **kwargs):
        H,N,C = self.oracle.dataset.preds.shape
        labels = self.oracle.labels
        one_hot = torch.nn.functional.one_hot(labels, num_classes=C).float()
        return one_hot