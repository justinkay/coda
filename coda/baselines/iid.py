import random
import torch
from coda.base import ModelSelector
from coda.beta import weighted_disagreement_candidates  # keep or move as needed


class IID(ModelSelector):
    def __init__(self, dataset, loss_fn):
        self.H, self.N, self.C = dataset.preds.shape
        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(self.N))
        self.dataset = dataset
        self.device = dataset.preds.device
        self.loss_fn = loss_fn

    def get_next_item_to_label(self):
        """
        Return (chosen_idx, selection_probability).
        """
        idx = random.choice(self.d_u_idxs)
        return idx, 1.0 / len(self.d_u_idxs)

    def add_label(self, chosen_idx, true_class, selection_prob=None):
        self.d_u_idxs.remove(chosen_idx)
        self.d_l_idxs.append(chosen_idx)
        self.d_l_ys.append(true_class)

    def get_risk_estimates(self):
        """
        Returns a tensor of shape (H,) giving each model's average loss
        on the labeled data so far.
        """
        risk = torch.zeros(self.H, device=self.device)
        if len(self.d_l_idxs) > 0:
            for idx, label in zip(self.d_l_idxs, self.d_l_ys):
                # shape of preds[:, idx, :] => (H, C)
                risk += self.loss_fn(
                    self.dataset.preds[:, idx, :], 
                    torch.tensor([label], device=self.device).expand(self.H)
                )
            risk /= len(self.d_l_idxs)
        return risk

    def get_best_model_prediction(self):
        """
        Returns the index of the model with the lowest average loss (risk).
        Ties are broken randomly.
        """
        risk = self.get_risk_estimates()
        best_model_risk, best_model_idx_pred = torch.min(risk, dim=0)
        ties = (risk == best_model_risk)
        if ties.sum() > 1:
            idxs = torch.nonzero(ties, as_tuple=True)[0]
            best_model_idx_pred = idxs[torch.randperm(len(idxs))[0]]
        return best_model_idx_pred
