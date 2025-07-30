import random
import torch

from coda.baselines.activetesting import ActiveTesting

class VMA(ActiveTesting):
    """
    Fom Matsuura & Hara (2023) "Variance Minimization for Active Model Selection"
    """
    def __init__(self, dataset, loss_fn):
        super().__init__(dataset, loss_fn)

    def get_next_item_to_label(self):
        """
        Return (chosen_idx, selection_probability).
        """
        device = self.dataset.device
        pi_y = self.surrogate.get_preds()
        pred_probs = self.dataset.preds # F.softmax(self.dataset.pred_logits, dim=2)
        pred_classes = pred_probs.argmax(dim=2)
        cols_items = torch.arange(self.N, device=device).unsqueeze(0).expand(self.H, self.N)
        y_star_probs = pi_y[ cols_items, pred_classes ]

        losses_all = 1.0 - y_star_probs
        losses = losses_all[:, self.d_u_idxs]  # shape (H, |D_U|)

        # Vectorized computation of pairwise variance minimization objective
        # Paper formula: Σ_{h'>h} |loss_h(x) - loss_h'(x)|
        
        # a) expand - shape (H,1,|D_U|) and shape (1,H,|D_U|)
        diff_3d = (losses.unsqueeze(0) - losses.unsqueeze(1)).abs()  
        # -> shape(H, H, |D_U|) where diff_3d[h', h, i] = |loss_h'(x_i) - loss_h(x_i)|

        # b) create an upper-triangular mask (h'>h to avoid double counting):
        mask = torch.triu(torch.ones(self.H, self.H, dtype=torch.bool, device=device), diagonal=1)
        # -> mask[h',h] = True if h'>h (upper triangular excluding diagonal)

        # c) apply the mask and sum across pairs => shape(|D_U|)
        masked_diffs = diff_3d[mask]  # Extract upper triangular elements
        # -> shape( (#pairs = H*(H-1)/2), |D_U| )
        acquisition_scores = masked_diffs.sum(dim=0)  # Sum over all pairs
        # -> shape(|D_U|) - final acquisition scores

        # normalize
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
