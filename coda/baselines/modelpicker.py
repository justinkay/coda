import torch
from coda.base import ModelSelector

from surrogates import Ensemble, WeightedEnsemble
import metrics


# best epsilons based on self-supervised grid search
# for ties, choose the median of equivalent epsilons
TASK_EPS = {
    # DomainNet
    'real_clipart': 0.36, #BEST => eps=0.360, Score=0.884
    'real_painting': 0.46, #BEST => eps=0.460, Score=0.942
    'real_sketch': 0.48, # BEST => eps=0.480, Score=0.899
    'sketch_real' : 0.49, #BEST => eps=0.490, Score=0.911
    'sketch_clipart' : 0.36, #BEST => eps=0.360, Score=0.916
    'sketch_painting' : 0.45, #BEST => eps=0.450, Score=0.929
    'clipart_painting' : 0.44, #BEST => eps=0.440, Score=0.919
    'clipart_real' : 0.37, #BEST => eps=0.370, Score=0.900
    'clipart_sketch': 0.47, #BEST => eps=0.470, Score=0.914
    'painting_sketch' : 0.49, #BEST => eps=0.490, Score=0.856
    'painting_real' : 0.36, #BEST => eps=0.360, Score=0.922
    'painting_clipart': 0.40, #BEST => eps=0.400, Score=0.902

    # WILDS
    'iwildcam': 0.49,       # BEST => eps=0.490, Score=0.816
    'civilcomments': 0.43,  # BEST => eps=0.430, Score=0.956
    'fmow': 0.38,           # BEST => eps=0.380, Score=0.727
    'camelyon': 0.35,       # BEST => eps=0.350, Score=0.902

    # MS-vision
    'imagenet_v2_matched-frequency': 0.48,  # BEST => eps=0.480, Score=0.857
    'cifar10_4070': 0.35,                   # BEST => eps=0.350, Score=0.722
    'cifar10_5592': 0.35,                   # BEST => eps=0.350, Score=0.905
    'pacs': 0.45,                           # BEST => eps=0.450, Score=0.940

    # GLUE
    'glue/cola': 0.37, # BEST => eps=0.370, Score=0.950
    'glue/mnli': 0.37, # BEST => eps=0.370, Score=0.916
    'glue/qnli': 0.4,  # BEST => eps=0.400, Score=0.942 
    'glue/qqp': 0.38,  # BEST => eps=0.380, Score=0.963 
    'glue/rte': 0.42,  # (all values tied) BEST => eps=0.350, Score=0.903 
    'glue/sst2':0.41,  # (most values tied) BEST => eps=0.350, Score=0.971

}


class ModelPicker(ModelSelector):
    """
    A ModelSelector that implements the 'ModelPicker' logic:
      - Maintains a posterior over model correctness using param 'gamma'
      - On each get_next_item_to_label call, picks the unlabeled item
        that yields minimal "conditional entropy" under that posterior
        (i.e., a re-implementation of the code snippet you showed).
    """

    def __init__(self, dataset, epsilon=0.46, prior_source="uniform", item_prior_source="uniform"):
        """
        dataset: your dataset object that presumably has:
           - dataset.pred_logits or dataset.predictions of shape (H, N, C) or (N,H)
           - dataset.oracle (if you store ground-truth somewhere)
        """
        self.dataset = dataset
        self.device = dataset.preds.device
        self.H, self.N, self.C = dataset.preds.shape

        self.epsilon = epsilon
        self.gamma = (1.0 - self.epsilon) / self.epsilon

        ensemble = Ensemble(dataset.preds)
        ensemble_preds = ensemble.get_preds() # (N, C)
        if prior_source == "ens-exp":
            pred_losses = metrics.simple_expected_error(ensemble_preds, dataset.preds).mean(dim=-1)
            self.posterior = 1 - pred_losses
        elif prior_source == "ens-01":
            pred_losses = metrics.simple_error(ensemble_preds, dataset.preds)
            self.posterior = 1 - pred_losses
        else:
            self.posterior = torch.ones(self.H, dtype=torch.float) / float(self.H)  # uniform init

        self.posterior = (self.posterior / self.posterior.sum()).to(self.dataset.device)

        self.item_prior_source = item_prior_source
        if item_prior_source == "ens":
            self.item_priors = ensemble_preds # TODO; what about for DS etc.
        elif item_prior_source == "none" or item_prior_source == "uniform" or item_prior_source == None: # uniform
            self.item_priors = torch.ones((self.N, self.C), device=self.device) * 1/self.C
        elif item_prior_source == "bma-adaptive":
            self.item_priors = self.get_bma_preds()
        else:
            raise NotImplemented

        # track how many times each model has predicted correctly on labeled items
        self.correct_counts = torch.zeros(self.H, dtype=torch.long, device=self.device)

        # labeled and unlabeled points
        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(dataset.preds.shape[1]))

        # because we start with a uniform prior, runs always start with a random choice
        self.stochastic = True

    def get_next_item_to_label(self):
        """
        Single-step approach:
          1) For each unlabeled item, compute 'entropy(i)' with 'compute_entropies(...)'
          2) pick item with minimal entropy, tie-broken randomly
          3) update self.posterior with that chosen item (simulate or wait until we know label?)
          4) return (chosen_idx, selection_probability)
        """
        device = self.dataset.device

        # We'll define 'predictions' as shape (N,H) => predicted class for item i by model h
        # If your dataset stores them differently, adapt accordingly:
        # e.g. pred_logits => shape (H,N,C). We'll do something like:
        #   pred_probs = F.softmax(self.dataset.pred_logits, dim=2) # => (H,N,C)
        #   predictions = pred_probs.argmax(dim=2).transpose(0,1)   # => (N,H)
        # if you already have dataset.predictions => (N,H), just use that directly
        # shape (H,N,C)
        pred_probs = self.dataset.preds # F.softmax(self.dataset.pred_logits, dim=2)
        # shape (H,N)
        pred_classes_hn = pred_probs.argmax(dim=2)
        # shape (N,H)
        predictions = pred_classes_hn.transpose(0,1).to(device)

        # predictions shape => (N,H). We'll gather only the unlabeled
        # => shape(|U|,H)
        preds_unlabeled = predictions[self.d_u_idxs]

        # We'll also define 'oracle' if needed. Typically you do that in 'add_label()'
        # but if we want to do a complete approach, we might store it. We'll skip here
        # since we only pick the next item. We'll do the "update posterior" once we know the label.

        # 1) compute entropies => shape(|U|)
        entropies = self.compute_entropies(preds_unlabeled, self.posterior, self.H, self.C, self.gamma)

        # 2) pick item with minimal entropy
        min_val = torch.min(entropies)
        # collect indices that have that value
        # shape (|indices with that min|)
        loc_i_stars = torch.nonzero(entropies == min_val).flatten()
        # pick one randomly among them
        i_star = loc_i_stars[torch.randint(len(loc_i_stars), (1,))].item()

        # This i_star is local index in [0..|U|-1], so let's map back to the global item idx
        chosen_idx = self.d_u_idxs[i_star]

        # 3) define selection_probability => we can define 1/|U| or do a soft approach
        selection_prob = 1.0 / float(len(self.d_u_idxs))

        # We do NOT do the posterior update here unless we have the actual label
        # We'll do that in 'add_label' once we get the ground truth from the user
        # So just return now

        return chosen_idx, selection_prob

    def add_label(self, chosen_idx, true_class, selection_prob=None):
        """
        Once we get the label from the oracle, we update the posterior
        and remove 'chosen_idx' from unlabeled set.
        """
        # remove from unlabeled
        self.d_u_idxs.remove(chosen_idx)

        # store the labeled info if desired
        self.d_l_idxs.append(chosen_idx)
        self.d_l_ys.append(true_class)

        # 1) gather predictions => shape(H,) for chosen_idx
        device = self.dataset.device
        if hasattr(self.dataset, "preds"):
            # same logic as above to get discrete predictions
            pred_probs = self.dataset.preds[:, chosen_idx, :] #F.softmax(self.dataset.pred_logits[:, chosen_idx, :], dim=1)
            # shape(H,C). Then argmax => shape(H,)
            pred_classes_h = pred_probs.argmax(dim=1)
        else:
            # or if we have dataset.predictions => shape(N,H),
            # then predictions[chosen_idx] => shape(H,)
            pred_classes_h = self.dataset.predictions[chosen_idx]

        # 2) Convert to numpy if you want or keep in torch
        # We'll do in torch for consistency:
        agreements_i = (pred_classes_h.to(device) == true_class)  # shape(H,)
        # convert bool to int
        agreements_i = agreements_i.long()

        # track correct predictions for accuracy-based model selection
        self.correct_counts += agreements_i

        # 3) update posterior
        # posterior => shape(H,).  We'll do the same formula as code snippet:
        # next_posterior = posterior * gamma^agreements
        # then normalize
        gamma_vec = (self.gamma ** agreements_i.float()).to(device)
        new_posterior = self.posterior * gamma_vec
        norm_const = new_posterior.sum()
        if norm_const < 1e-12:
            # fallback to uniform if everything is 0
            new_posterior = torch.ones_like(self.posterior) / float(self.H)
        else:
            new_posterior = new_posterior / norm_const
        self.posterior = new_posterior

    def compute_entropies(self, predictions_unlabeled, posterior, num_models, num_classes, gamma):
        """
        Vectorized version that computes the 'conditional entropies' for each unlabeled item,
        exactly as in the original 'compute_entropies' method, but in PyTorch.

        predictions_unlabeled: shape(|U|, H) => predicted class for each unlabeled item across H models
        posterior: shape(H,) => the current model posterior
        num_models: H
        num_classes: C
        gamma: (1 - epsilon)/epsilon
        returns shape(|U|,) => the 'entropy' for each item
        """
        device = predictions_unlabeled.device
        # number of unlabeled items
        num_unlabeled = predictions_unlabeled.shape[0]

        # replicate posterior => shape(|U|, H)
        # so each row is a copy of the posterior
        posteriors_replicated = posterior.unsqueeze(0).expand(num_unlabeled, num_models)

        entropies = torch.zeros(num_unlabeled, dtype=torch.float, device=device)

        if self.item_prior_source == "bma-adaptive":
            self.item_priors = self.get_bma_preds()

        # We'll do the same summation over c in [num_classes]:
        for c in range(num_classes):
            # agreements_c => shape(|U|, H): 1 if model h predicts class c for item i
            agreements_c = (predictions_unlabeled == c).long()

            # new_posteriors_c => shape(|U|, H)
            # multiply each posterior entry by gamma^(1) if it agrees, else gamma^(0)=1
            # i.e. new_posteriors_c[i,h] = posteriors_replicated[i,h]* (gamma^agreements_c[i,h])
            pow_terms = (gamma**(agreements_c.float()))
            new_posteriors_c = posteriors_replicated * pow_terms

            # normalization => sum over h => shape(|U|)
            norm_const = new_posteriors_c.sum(dim=1, keepdim=True).clamp_min(1e-12)
            # shape(|U|, 1)
            new_normed = new_posteriors_c / norm_const  # shape(|U|, H)

            # compute the (rowwise) entropies with base=2 => use log2:
            # stats.entropy => - sum p log p. We'll do in torch:
            p_clamped = new_normed.clamp_min(1e-12)
            conditional_entropies = - (p_clamped * torch.log2(p_clamped)).sum(dim=1)
            # shape(|U|,)

            # Now weight each item's conditional entropy by its
            # probability of belonging to class c.
            # item_class_weights => shape(|U|, C)
            # item_class_weights[:, c] => shape(|U|,
            entropies += self.item_priors[self.d_u_idxs, c] * conditional_entropies

        return entropies

    def get_best_model_prediction(self):
        """Return the index of the model with the highest accuracy on labeled data so far. Ties are broken randomly."""
        if len(self.d_l_idxs) == 0:
            # no labels yet -> pick randomly
            return torch.randint(self.H, (1,)).item()
        max_acc = torch.max(self.correct_counts)
        ties = torch.nonzero(self.correct_counts == max_acc, as_tuple=True)[0]
        idx = ties[torch.randint(len(ties), (1,))]
        return idx.item()
    
    def get_bma_preds(self):
        prob_best = self.posterior
        ensemble = WeightedEnsemble(self.dataset.preds)
        preds = ensemble.get_preds(weights=prob_best)
        return preds