import torch
import torch.nn.functional as F

from ams.iid import IID


class Uncertainty(IID):
    def __init__(self, dataset, loss_fn, prefilter_fn=None, prefilter_n=500):
        self.H, self.N, self.C = dataset.pred_logits.shape
        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(self.N))
        self.dataset = dataset
        self.device = dataset.pred_logits.device
        self.loss_fn = loss_fn

    def get_next_item_to_label(self):
        pred_probs = F.softmax(self.dataset.pred_logits, dim=2)
        mean_pred_probs = pred_probs.mean(dim=0)
        epsilon = 1e-8
        entropy_per_data_point = -torch.sum(mean_pred_probs * torch.log(mean_pred_probs + epsilon), dim=-1)
        entropy_per_unlabeled_data_point = entropy_per_data_point[self.d_u_idxs]
        chosen_q, chosen_idx_local = torch.max(entropy_per_unlabeled_data_point, dim=0)
        chosen_idx_global = self.d_u_idxs[chosen_idx_local]
        return chosen_idx_global, chosen_q