import random
import torch
from ams.base import ModelSelector
from ams.bb import weighted_disagreement_candidates  # keep or move as needed

class IID(ModelSelector):
    def __init__(self, dataset, loss_fn, prefilter_fn=None, prefilter_n=500):
        self.H, self.N, self.C = dataset.preds.shape
        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(self.N))
        self.dataset = dataset
        self.device = dataset.preds.device
        self.loss_fn = loss_fn
        self.prefilter_fn = prefilter_fn
        self.prefilter_n = prefilter_n
        self.prefilter_h = 50

    def get_next_item_to_label(self):
        """
        Return (chosen_idx, selection_probability).
        """
        # If we have fewer unlabeled items than prefilter_n,
        # or prefilter is None => pick uniformly from all unlabeled.
        # if len(self.d_u_idxs) <= self.prefilter_n or self.prefilter_fn is None:
        idx = random.choice(self.d_u_idxs)
        return idx, 1.0 / len(self.d_u_idxs)
        # else:
        #     print("HELP IID", len(self.d_u_idxs), self.prefilter_n, self.prefilter_fn)

        # Otherwise, apply the prefilter
        # if self.prefilter_fn == 'disagreement':
        #     # 1) compute some "prob_best" or "importance" for each model
        #     #    We'll hack something like "prob_best ~ 1 - normalized risk"
        #     risks = self.get_risk_estimates()                     # shape (H,)
        #     prob_best = (risks / risks.sum()).clamp_min(1e-9)  # shape (H,) – just a toy example

        #     # 2) gather integer predictions => shape (H,N) => transpose => shape(N,H)
        #     pred_classes = torch.argmax(self.dataset.preds, dim=-1).T

        #     # 3) run the disagreement filter on *only the unlabeled subset*
        #     #    so we do pred_classes[self.d_u_idxs], shape (M, H)
        #     #    weighted_disagreement_candidates wants shape (N, H),
        #     #    so we pass that in as the first argument,
        #     #    plus "prob_best" of shape (H,) for the second argument
        #     subset_local = weighted_disagreement_candidates(
        #         pred_classes[self.d_u_idxs],       # shape (M,H)
        #         prob_best, 
        #         k=min(self.prefilter_h, self.H),
        #         n=min(self.prefilter_n, len(self.d_u_idxs))
        #     )
        #     # subset_local are local indices in [0..M-1]
        #     # Convert them back to global indices
        #     subset_global = [self.d_u_idxs[i] for i in subset_local]

        #     # 4) Pick uniformly among that subset
        #     chosen_idx = random.choice(subset_global)
        #     chosen_q   = 1.0 / len(subset_global)
        #     return chosen_idx, chosen_q
        # else:
        #     raise NotImplementedError("Only 'disagreement' prefilter is supported.")

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
