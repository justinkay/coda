import torch

from coda.baselines.iid import IID


class Uncertainty(IID):
    """
    Uncertainty sampling for active model selection.
    
    Adapts classic uncertainty sampling from active learning to model selection context.
    Selects data points where the ensemble (committee) of models has highest disagreement.
    
    Algorithm:
    1. Compute mean prediction across all models: p_mean = (1/H) * Σ_h p_h(x)
    2. Calculate entropy: H(p_mean) = -Σ_c p_mean(c) * log(p_mean(c))
    3. Select point with maximum entropy (highest uncertainty)
    
    This is a non-adaptive baseline - acquisition function doesn't change based on 
    previously labeled points.
    
    Reference: Committee-based uncertainty sampling (Dagan & Engelson, 1995)
    """
    def __init__(self, dataset, loss_fn):
        super().__init__(dataset, loss_fn)
        self.stochastic = False 

    def get_next_item_to_label(self):
        mean_pred_probs = self.dataset.preds.mean(dim=0) # average over all models
        epsilon = 1e-8
        entropy_per_data_point = -torch.sum(mean_pred_probs * torch.log(mean_pred_probs + epsilon), dim=-1)
        entropy_per_unlabeled_data_point = entropy_per_data_point[self.d_u_idxs]
        chosen_q, chosen_idx_local = torch.max(entropy_per_unlabeled_data_point, dim=0)
        ties = (entropy_per_unlabeled_data_point == chosen_q)
        if ties.sum() > 1:
            self.stochastic = True
            idxs = torch.nonzero(ties, as_tuple=True)[0]
            chosen_idx_local = idxs[torch.randperm(len(idxs))[0]]
        chosen_idx_global = self.d_u_idxs[chosen_idx_local]
        return chosen_idx_global, chosen_q.item()