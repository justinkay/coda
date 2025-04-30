import random
import torch
import torch.nn.functional as F

from coda.baselines.activetesting import ActiveTesting

class VMA(ActiveTesting):
    def __init__(self, dataset, loss_fn):
        super().__init__(dataset, loss_fn)

    # vectorized
    def get_next_item_to_label(self):
        """
        Return (chosen_idx, selection_probability).
        """
        device = self.dataset.device

        # 1) Surrogate probabilities for each item => shape (N, C)
        pi_y = self.surrogate.get_preds()   # e.g. pi_y[i,c] = prob that item i is class c

        # 2) Main models' predicted probs => shape (H, N, C)
        pred_probs = self.dataset.preds # F.softmax(self.dataset.pred_logits, dim=2)
        # Argmax => shape (H, N): model h picks its best class for each item
        pred_classes = pred_probs.argmax(dim=2)

        # 3) Surrogate's prob for those predicted classes => shape (H, N)
        #    y_star_probs[h, i] = pi_y[i, predicted_class_of_model_h]
        #    We'll do a gather approach:
        rows_h     = torch.arange(self.H, device=device).unsqueeze(-1).expand(self.H, self.N)
        cols_items = torch.arange(self.N, device=device).unsqueeze(0).expand(self.H, self.N)
        # pred_classes[h,i] is the class chosen by model h for item i
        # pi_y[i,c], so to gather we do pi_y[ cols_items, pred_classes[h,i] ]
        # But let's keep it simple using advanced indexing:
        y_star_probs = pi_y[ cols_items, pred_classes ]

        # 4) "Losses" = 1 - y_star_probs => shape (H, N)
        #    Then restrict to unlabeled items => shape (H, |D_U|)
        losses_all = 1.0 - y_star_probs
        losses = losses_all[:, self.d_u_idxs]  # shape (H, |D_U|)

        # 5) Instead of nested loops over (hprime, h), do a vectorized approach:
        #    We want sum_{hprime > h} abs(losses[h] - losses[hprime]) for each item.
        #
        #    We'll do the pairwise differences in one operation:
        #    diff_3d[hprime,h,:] = abs( losses[hprime,:] - losses[h,:] )
        #    then we sum over all pairs hprime>h.
        #
        # a) expand => shape (H,1,|D_U|) and shape (1,H,|D_U|)
        diff_3d = (losses.unsqueeze(0) - losses.unsqueeze(1)).abs()  
        # => shape(H, H, |D_U|)

        # b) create an upper-triangular mask (hprime>h):
        mask = torch.triu(torch.ones(self.H, self.H, dtype=torch.bool, device=device), diagonal=1)
        # => mask[hprime,h] = True if hprime>h

        # c) apply the mask => shape(#pairs, |D_U|)
        #    then sum across the #pairs dimension => shape(|D_U|)
        masked_diffs = diff_3d[mask]
        # => shape( (#pairs = H*(H-1)/2), |D_U| )
        acquisition_scores = masked_diffs.sum(dim=0)
        # => shape(|D_U|)

        # 6) Normalize => shape(|D_U|)
        total = acquisition_scores.sum()
        if total < 1e-12:
            # fallback: if all zero, pick uniformly
            chosen_idx = random.choice(self.d_u_idxs)
            chosen_q = 1.0 / len(self.d_u_idxs)
        else:
            acquisition_scores /= total
            # 7) pick item via random.choices
            chosen_idx = random.choices(
                self.d_u_idxs,
                weights=acquisition_scores.cpu().tolist(),
                k=1
            )[0]
            # to get the final probability for reference:
            chosen_local = self.d_u_idxs.index(chosen_idx)
            chosen_q = acquisition_scores[chosen_local].item()

        return chosen_idx, chosen_q
