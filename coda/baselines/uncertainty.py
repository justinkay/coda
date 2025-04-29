import torch

from coda.baselines.iid import IID


class Uncertainty(IID):
    def __init__(self, dataset, loss_fn):
        super().__init__(dataset, loss_fn)

    def get_next_item_to_label(self):
        mean_pred_probs = self.dataset.preds.mean(dim=0) # average over all models
        epsilon = 1e-8
        entropy_per_data_point = -torch.sum(mean_pred_probs * torch.log(mean_pred_probs + epsilon), dim=-1)
        entropy_per_unlabeled_data_point = entropy_per_data_point[self.d_u_idxs]
        chosen_q, chosen_idx_local = torch.max(entropy_per_unlabeled_data_point, dim=0)
        chosen_idx_global = self.d_u_idxs[chosen_idx_local]
        return chosen_idx_global, chosen_q