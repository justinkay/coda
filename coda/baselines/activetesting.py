import random
import torch
import torch.nn.functional as F

from coda.baselines.iid import IID
from coda.util import Ensemble


class ActiveTesting(IID):
    """
    From Kossen et al. (2021).
    Surrogate is an ensemble of all model candidates. 
    Acq fn is expected loss naively summed over all model candidates.
    """
    def __init__(self, dataset, loss_fn):
        super().__init__(dataset, loss_fn)

        self.surrogate = Ensemble(dataset.preds)

        # Actively sampled points
        self.M = 0  # Number of sampled points
        self.losses = []  # True losses for each model - shape (H, M)
        self.qs = []  # Sampling probabilities for each point - shape (M,)

        # always stochastic by definition
        self.stochastic = True

    def get_next_item_to_label(self):
        """
        Return (chosen_idx, selection_probability).
        """
        # Compute the surrogate's probabilities
        pi_y = self.surrogate.get_preds()

        # Compute the main models' predicted classes
        pred_probs = self.dataset.preds
        pred_classes = pred_probs.argmax(dim=2)

        # Get the surrogate's probability for the predicted classes
        y_star_probs = pi_y[torch.arange(pi_y.shape[0]), pred_classes]
        
        # Sum acquisition scores across all models for model selection
        acquisition_scores = 1 - y_star_probs # (H, N) - loss for each model on each point
        acquisition_scores = acquisition_scores.sum(dim=0)[self.d_u_idxs]  # Sum across models
        acquisition_scores /= acquisition_scores.sum()

        chosen_idx = random.choices(self.d_u_idxs, weights=acquisition_scores.cpu().numpy().tolist())[0]
        chosen_q = acquisition_scores[self.d_u_idxs.index(chosen_idx)].item()
        
        return chosen_idx, chosen_q

    def get_vs(self):
        """
        Compute the LURE weights (v_m) for each sampled point based on the current state.
        This is called after adding a new observation to update the weights.
        
        LURE formula from Farquhar et al. (2021):
        v_m = 1 + (N-M)/(N-m) * (1/((N-m+1)*q_m) - 1)
        where m is 1-indexed in the formula
        """
        vs = []
        for m in range(self.M):
            q = self.qs[m]
            m_idx = m + 1  # Convert to 1-indexed for formula
            v = 1 + ( (self.N - self.M) / (self.N - m_idx) ) * (1 / ((self.N - m_idx + 1) * q) - 1)
            vs.append(v)
        return vs

    def get_lure_risks_and_vars(self):
        # Stack losses to shape (H, M)
        losses = torch.stack(self.losses, dim=1).view(self.H, -1)  # Shape (H, M)
        vs = torch.tensor(self.get_vs(), device=self.device).unsqueeze(0)  # Shape (1, M)

        # Compute weighted losses: w_m = v_m * L_m
        weighted_losses = vs * losses  # Shape (H, M)

        # Compute LURE estimates (mean of weighted losses)
        lure_estimates = weighted_losses.mean(dim=1)  # Shape (H,)

        # Compute sample variance of weighted losses (Var[w_m])
        sample_variance = weighted_losses.var(dim=1, unbiased=True)  # Shape (H,)

        # Compute variance of the LURE estimator (Var[hat{R}_{LURE}] = Var[w_m] / M)
        variance_lure = sample_variance / self.M  # Shape (H,)

        # print("self losses stack [0]", torch.stack(self.losses, dim=1)[0].shape, torch.stack(self.losses, dim=1)[0])
        # print("vs", vs)
        # print("weighted losses 0", weighted_losses[0])

        return lure_estimates, variance_lure

    def add_label(self, chosen_idx, true_class, selection_prob=None):
        super().add_label(chosen_idx, true_class, selection_prob)
        loss = self.loss_fn(self.dataset.preds[:, chosen_idx, :], torch.tensor([true_class], device=self.device).repeat(self.H), reduction='none')
        self.losses.append(loss)
        self.qs.append(selection_prob)
        self.M += 1

    def get_risk_estimates(self):
        """
        Returns a tensor of shape (H,) giving each model's average loss
        on the labeled data so far.
        """
        lure_risks, lure_vars = self.get_lure_risks_and_vars()
        return lure_risks

    def get_best_model_prediction(self):
        """
        Returns the index of the model with the lowest average loss (risk).
        Ties are broken randomly.
        """
        if len(self.losses):
            risk = self.get_risk_estimates()
            best_model_risk, best_model_idx_pred = torch.min(risk, dim=0)
            ties = (risk == best_model_risk)
            if ties.sum() > 1:
                idxs = torch.nonzero(ties, as_tuple=True)[0]
                best_model_idx_pred = idxs[torch.randperm(len(idxs))[0]]
            return best_model_idx_pred
        else: 
            idxs = torch.arange(self.surrogate.preds.shape[0], device=self.surrogate.device)
            return random.choice(idxs)