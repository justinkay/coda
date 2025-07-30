import torch
from coda.base import ModelSelector


TASK_EPS = {
    # from original paper
    'imagenet_v2_matched-frequency': 0.48,
    'cifar10_4070': 0.47,
    'cifar10_5592': 0.47,
    'pacs': 0.45,
    'glue/cola': 0.45,
    'glue/mnli': 0.43,
    'glue/qnli': 0.44,
    'glue/qqp': 0.47,
    'glue/rte': 0.39,
    'glue/sst2': 0.36,
    
    # from our reproduction of their gridsearch
    'real_clipart': 0.42,
    'real_painting': 0.35,
    'real_sketch': 0.45,
    'sketch_real': 0.35,
    'sketch_clipart': 0.35,
    'sketch_painting': 0.37,
    'clipart_painting': 0.45,
    'clipart_real': 0.45,
    'clipart_sketch': 0.43,
    'painting_sketch': 0.39,
    'painting_real': 0.44,
    'painting_clipart': 0.39,
    'iwildcam': 0.49,
    'civilcomments': 0.46,
    'fmow': 0.44,
    'camelyon': 0.47,
}

class ModelPicker(ModelSelector):
    def __init__(self, dataset, epsilon=0.46):
        self.dataset = dataset
        self.device = dataset.preds.device
        self.H, self.N, self.C = dataset.preds.shape

        preds = self.dataset.preds.argmax(dim=2).transpose(0, 1)  # (N, H)
        first_col = preds[:, [0]]
        self._disagreement_mask = (preds != first_col).any(dim=1)

        self.epsilon = float(epsilon)
        self.gamma = (1.0 - self.epsilon) / self.epsilon
        self.posterior = torch.ones(self.H, device=self.device) / self.H

        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(self.N))
        self.correct_counts = torch.zeros(self.H, dtype=torch.long, device=self.device)

        self.stochastic = True

    def get_next_item_to_label(self):
        preds = self.dataset.preds.argmax(dim=2).transpose(0, 1)  # (N, H)
        preds_unlabeled = preds[self.d_u_idxs]

        mask = self._disagreement_mask[self.d_u_idxs]
        entropies = self.compute_entropies(preds_unlabeled, self.posterior, self.H, self.C, self.gamma)
        if mask.any():
            entropies = entropies.clone()
            entropies[~mask] = float('inf')

        min_val = torch.min(entropies)
        loc_i_stars = torch.nonzero(entropies == min_val).flatten()
        i_star = loc_i_stars[torch.randint(len(loc_i_stars), (1,))].item()
        chosen_idx = self.d_u_idxs[i_star]
        return chosen_idx, 1.0 / float(len(self.d_u_idxs))

    def compute_entropies(self, predictions_unlabeled, posterior, num_models, num_classes, gamma):
        num_unlabeled = predictions_unlabeled.shape[0]
        posteriors_replicated = posterior.unsqueeze(0).expand(num_unlabeled, num_models)
        entropies = torch.zeros(num_unlabeled, device=self.device)
        for c in range(num_classes):
            agreements = (predictions_unlabeled == c).float()
            new_posteriors = posteriors_replicated * (gamma ** agreements)
            norm_const = new_posteriors.sum(dim=1, keepdim=True)
            new_normalized = new_posteriors / norm_const
            p = new_normalized.clamp(min=1e-12)
            conditional = -(p * torch.log2(p)).sum(dim=1)
            entropies += conditional / num_classes
        return entropies

    def add_label(self, chosen_idx, true_class, selection_prob=None):
        self.d_u_idxs.remove(chosen_idx)
        self.d_l_idxs.append(chosen_idx)
        self.d_l_ys.append(true_class)

        preds = self.dataset.preds[:, chosen_idx].argmax(dim=1)
        self.correct_counts += (preds == true_class).long()

        self.posterior = self.update_posterior(self.posterior, preds, true_class, self.gamma)

    def update_posterior(self, posterior, predictions_i, oracle_i, gamma):
        agreements = (predictions_i == oracle_i).float()
        next_post = posterior * (gamma ** agreements)
        next_post = next_post / next_post.sum()
        return next_post

    def get_best_model_prediction(self):
        if len(self.d_l_idxs) == 0:
            return torch.randint(self.H, (1,), device=self.device).item()
        max_acc = torch.max(self.correct_counts)
        ties = torch.nonzero(self.correct_counts == max_acc).flatten()
        idx = ties[torch.randint(len(ties), (1,), device=self.device)].item()
        return idx
