import random
import torch
import torch.nn.functional as F

from coda.base import ModelSelector
from coda.baselines.iid import IID
from surrogates import Ensemble

class ActiveTesting(IID):
    def __init__(self, dataset, loss_fn, prefilter_fn=None, prefilter_n=500):
        self.H, self.N, self.C = dataset.pred_logits.shape
        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(self.N))
        self.dataset = dataset
        self.device = dataset.pred_logits.device
        self.loss_fn = loss_fn
        self.prefilter_fn = prefilter_fn
        self.prefilter_n = prefilter_n
        self.prefilter_h = 50

        self.surrogate = Ensemble(dataset.pred_logits)

        # Actively sampled points
        self.M = 0  # Number of sampled points
        self.losses = []  # True losses for each model - shape (H, M)
        self.qs = []  # Sampling probabilities for each point - shape (M,)


    def get_next_item_to_label(self):
        """
        Return (chosen_idx, selection_probability).
        """
        # Compute the surrogate's probabilities
        pi_y = self.surrogate.get_preds()

        # Compute the main models' predicted classes
        pred_probs = F.softmax(self.dataset.pred_logits, dim=2)
        pred_classes = pred_probs.argmax(dim=2)

        # Get the surrogate's probability for the predicted classes
        y_star_probs = pi_y[torch.arange(pi_y.shape[0]), pred_classes]
        
        # Compute the acquisition score
        acquisition_scores = 1 - y_star_probs # (H, N)
        acquisition_scores = acquisition_scores.sum(dim=0)[self.d_u_idxs]
        acquisition_scores /= acquisition_scores.sum()

        chosen_idx = random.choices(self.d_u_idxs, weights=acquisition_scores.cpu().numpy().tolist())[0]
        chosen_q = acquisition_scores[self.d_u_idxs.index(chosen_idx)]
        
        return chosen_idx, chosen_q

    def get_vs(self):
        """
        Compute the LURE weights (v_m) for each sampled point based on the current state.
        This is called after adding a new observation to update the weights.
        """
        vs = []
        for m in range(self.M):
            q = self.qs[m]
            m = m + 1 # M is 1-indexed when computing v
            v = 1 + ( (self.N - self.M) / (self.N - m) ) * (1 / ((self.N - m + 1) * q) - 1)
            vs.append(v)
        return vs

    def get_lure_risks_and_vars(self):
        # Stack losses to shape (H, M)
        losses = torch.stack(self.losses, dim=1).view(self.H, -1)  # Shape (H, M)
        vs = torch.tensor(self.get_vs(), device=self.device).unsqueeze(0)  # Shape (1, M)

        # Compute weighted losses: w_m = v_m * L_m
        weighted_losses = vs * losses  # Shape (H, M)
        # TESTING: IGNORE VS
        # weighted_losses = losses

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
        loss = self.loss_fn(self.dataset.pred_logits[:, chosen_idx, :], torch.tensor([true_class], device=self.device).repeat(self.H), reduction='none')
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
        risk = self.get_risk_estimates()
        best_model_risk, best_model_idx_pred = torch.min(risk, dim=0)
        ties = (risk == best_model_risk)
        if ties.sum() > 1:
            idxs = torch.nonzero(ties, as_tuple=True)[0]
            best_model_idx_pred = idxs[torch.randperm(len(idxs))[0]]
        return best_model_idx_pred
