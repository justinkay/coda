import torch
from torch.distributions import Beta
import torch.nn.functional as F
from tqdm import tqdm
import random
import mlflow

from coda.base import ModelSelector
from coda.util import Ensemble, distribution_entropy, _check, _check_prob, plot_bar

_DEBUG = True       # check for valid PDFs, infs, NaNs
_DEBUG_VIZ = False  # plot PBest and EIG every iter


def dirichlet_to_beta(alpha_dirichlet: torch.Tensor):
    """
    Get parameters for beta distributions representing the diagonal.
    Args:
        alpha_dirichlet: shape (..., H, C, C)
    Returns:
        alpha_cc, beta_cc: shape (..., H, C)
    """
    C = alpha_dirichlet.shape[-1]
    alpha_cc = alpha_dirichlet[..., torch.arange(C), torch.arange(C)]
    beta_cc  = alpha_dirichlet.sum(dim=-1) - alpha_cc
    return alpha_cc, beta_cc


def create_confusion_matrices(true_labels: torch.Tensor,
                              model_predictions: torch.Tensor,
                              mode='hard') -> torch.Tensor:
    H, N, C = model_predictions.shape
    dev = model_predictions.device
    true_one_hot = F.one_hot(true_labels, C).float().to(dev)

    if mode == 'hard':
        preds = F.one_hot(model_predictions.argmax(-1), C).float()
    elif mode == 'soft':
        preds = model_predictions
    else:
        raise ValueError(mode)

    conf = torch.einsum('nc, hnj -> hcj', true_one_hot, preds)
    return conf / conf.sum(-1, keepdim=True).clamp_min(1e-6)


def initialize_dirichlets(soft_confusion: torch.Tensor,
                          prior_strength: float,
                          base_strength: float = 1.0,
                          base_prior: str = "diag",
                          upweight_binary: bool = False) -> torch.Tensor:
    H, C, _ = soft_confusion.shape
    if base_prior == "diag":
        if C == 2 and upweight_binary:
            base = torch.full((C, C), base_strength * 3 / 8,
                              dtype=soft_confusion.dtype,
                              device=soft_confusion.device)
            base.fill_diagonal_(base_strength * 5 / 8)
        else:
            # base = torch.full((C, C), base_strength / (C - 1),
            #                   dtype=soft_confusion.dtype,
            #                   device=soft_confusion.device)
            # base.fill_diagonal_(base_strength)
            # TESTING WHAT IF WE DO THIS
            base = torch.full((C, C), 1.0 / (C - 1),
                              dtype=soft_confusion.dtype,
                              device=soft_confusion.device)
            base.fill_diagonal_(1.0)
            
    elif base_prior == "uniform":
        base = torch.full((C, C), 2 * base_strength / C,
                          dtype=soft_confusion.dtype,
                          device=soft_confusion.device)
    elif base_prior == "empirical-bayes":
        base = 2 * soft_confusion.mean(0) + 1e-5
    base = base.unsqueeze(0).expand(H, C, C)

    return base + prior_strength * soft_confusion


def batch_update_dirichlet_for_item(dirichlet_alphas: torch.Tensor,
                                    classifier_preds: torch.Tensor,
                                    update_weight: float = 1.0) -> torch.Tensor:
    N, H, C = classifier_preds.shape
    updated = dirichlet_alphas[None, None].expand(N, C, H, C, C).clone()
    updates = classifier_preds[:, None].expand(-1, C, -1, -1) * update_weight
    for c in range(C):
        updated[:, c, :, c, :] += updates[:, c, :, :]
    return updated


def compute_p_best_beta_batched(alpha_batch: torch.Tensor,
                                beta_batch:  torch.Tensor,
                                num_points: int = 256,
                                eps: float = 1e-30,
                                chunk_size: int = None) -> torch.Tensor:
    device = alpha_batch.device
    if alpha_batch.ndim == 2:
        alpha_batch = alpha_batch[None]
        beta_batch = beta_batch[None]
        single_item = True
    else:
        single_item = False

    # N, C, H = alpha_batch.shape
    N = alpha_batch.shape[0]
    C, H = alpha_batch.shape[-2:]

    chunk_size = chunk_size or N
    x = torch.linspace(1e-6, 1 - 1e-6, num_points, device=device).unsqueeze(-1) # P×1
    # prob_out = torch.zeros(N, C, H, device=device)
    prob_out = torch.zeros_like(alpha_batch)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        a_flat = alpha_batch[start:end].reshape(-1, H)          # B_*C_*C × H
        b_flat = beta_batch[start:end].reshape(-1, H)
        # print("a flat", a_flat.shape)
        # print("b flat", b_flat.shape)
        
        logpdf = Beta(a_flat.reshape(-1), b_flat.reshape(-1)).log_prob(x)
        pdf = logpdf.exp().T.reshape(-1, H, num_points)         # B_*C_*C × H × P
        if _DEBUG: _check(pdf, "pdf")

        cdf = torch.zeros_like(pdf)
        for j in range(1, num_points):
            dx = x[j] - x[j-1]
            cdf[:, :, j] = cdf[:, :, j-1] + 0.5*(pdf[:, :, j] + pdf[:, :, j-1])*dx
        if _DEBUG: _check(cdf, "cdf")

        log_cdf = torch.log(cdf.clamp_min(eps))
        prod_excl = torch.exp(log_cdf.sum(1, keepdim=True) - log_cdf)
        integrand = pdf * prod_excl
        if _DEBUG: _check(integrand, "integrand")

        prob = torch.trapz(integrand, x.squeeze(), dim=2)
        if _DEBUG: _check(prob, "Pbest(beta)") 

        prob = prob / prob.sum(-1, keepdim=True).clamp_min(eps)
        if _DEBUG: _check_prob(prob, "Pbest(beta) normalized")

        prob_out[start:end] = prob.reshape(alpha_batch[start:end].shape)

    return prob_out[0] if single_item else prob_out


def p_best_row_mixture_batched(updated_dirichlet: torch.Tensor,
                                pi_hat: torch.Tensor,
                                num_points: int = 256) -> torch.Tensor:
    """
    Args:
        updated_dirichlet: (B_, C_, H, C, C)
        pi: (C,)
    where:
        B_ and C_ are additional dimensions that can be used for hypothetical item and hypothetical class updates, respectively
            (i.e. all operations are broadcast over B_ and C_)
        pi is the marginal class distribution P(class=C) over the entire dataset
    
    Returns:
        prob_best: (B_, C_, H),  P(h is best | C_, B_)
    """
    C = updated_dirichlet.shape[-1]

    # α_cc
    diag_full = torch.diagonal(updated_dirichlet, dim1=-2, dim2=-1) # (B_, C_, H, C)
    new_order = list(range(diag_full.ndim - 2)) + [diag_full.ndim - 1, diag_full.ndim - 2]
    alpha_cc = torch.permute(diag_full, new_order) # (B_, C_, C, H)

    # β_cc
    row_sum  = updated_dirichlet.sum(-1) # (B_, C_, H, C)
    beta_cc = torch.permute(row_sum, new_order) - alpha_cc # (B_, C_, C, H)

    # P(h is best | row c)
    prob_best_b_c_ch = compute_p_best_beta_batched(alpha_cc, beta_cc, num_points=num_points) # (B_,C_,C,H)
    
    # Convert conditional to marginal probabilities using pi
    # expected P(best | item b) = Σ_c expected P(best | item b, class=c) * P(class=c)
    marginal_probs = (prob_best_b_c_ch * pi_hat.view(1, C, 1)).sum(-2)  # (B_, C_, H)

    return marginal_probs


def old_eig_dirichlet_batched(dirichlet_alphas: torch.Tensor,
                          classifier_preds: torch.Tensor,
                          pi_hat: torch.Tensor,     # marginal
                          pi_hat_xi: torch.Tensor,  # per item
                          candidate_ids: list[int],
                          chunk_size: int = 100,
                          update_weight: float = 1.0,
                          num_points: int = 256) -> torch.Tensor:

    device   = dirichlet_alphas.device
    _, H, C  = classifier_preds.shape

    candidates = torch.tensor(candidate_ids, device=device)

    # compute current entropy
    H, C_dir, _ = dirichlet_alphas.shape
    expanded = dirichlet_alphas.unsqueeze(0).unsqueeze(0).expand(1, 1, H, C_dir, C_dir) # get it into shape p_best expects - dummy B_ and C_ dims
    current_probs = p_best_row_mixture_batched(expanded, pi_hat, num_points=num_points).squeeze()  # (H,)
    H_current = distribution_entropy(current_probs)

    eig_chunks = []
    for s in tqdm(range(0, len(candidates), chunk_size)):
        ids   = candidates[s:s + chunk_size] # (B,)
        preds = F.one_hot(classifier_preds[ids].argmax(-1), C).float()  # (B,H,C)

        # all hypothetical updates at once
        updated = batch_update_dirichlet_for_item(dirichlet_alphas, preds, update_weight) # (B,C,H,C,C)
        updated_probs = p_best_row_mixture_batched(updated, pi_hat, num_points) # (B,C,H)

        # entropy for each hypothetical outcome
        # weighted by probability of that outcome according to consensus votes (pi_hat_xi)
        pi_xi = pi_hat_xi[ids]
        p_clamped = updated_probs.clamp_min(1e-12)
        H_updated = (-(p_clamped * p_clamped.log2()).sum(-1) * pi_xi) # (B,C)
        H_updated = H_updated.sum(-1) # (B,)
        # print("H updated, H_current", H_updated, H_current)

        # expected IG  = H_current − E[H_updated]
        eig_chunk = H_current - H_updated                         # (B,)
        eig_chunks.append(eig_chunk)                              # (B,)

    return torch.cat(eig_chunks)                                  # (N_candidates,)

def batch_update_beta(selector, # selector.dirichlets: (H,C,C)
                      preds,    # (B, H)
                      update_weight=1.0
                      ): 
    B, H = preds.shape
    C = selector.dirichlets.shape[-1]
    alpha_cc_before, beta_cc_before = dirichlet_to_beta(selector.dirichlets) # (H, C)

    pred_classes = preds.unsqueeze(1).expand(B,C,H)
    class_range  = torch.arange(C, device=alpha_cc_before.device).unsqueeze(1).expand(B,C,H)
    eq_mask = (pred_classes == class_range) # B,C,H
    eq_mask = eq_mask.permute(0,2,1) # B,H,C
    # print("eq_mask", eq_mask.shape)

    alpha_batch = alpha_cc_before.expand(B, H, C).clone()
    beta_batch = beta_cc_before.expand(B, H, C).clone()
    alpha_batch[eq_mask] += 1.0 * update_weight
    beta_batch[~eq_mask]  += 1.0 * update_weight

    return alpha_batch, beta_batch # (B, H, C), (B, H, C)


def eig_dirichlet_batched(selector,
                          candidate_ids: list[int],
                          chunk_size: int = 100,
                          update_weight: float = 1.0,
                          num_points: int = 256) -> torch.Tensor:
    
    classifier_preds = selector.dataset.preds.permute(1, 0, 2)
    candidates = torch.tensor(candidate_ids, device=classifier_preds.device)
    N, H, C = classifier_preds.shape

    # compute current pbest per row
    dirichlets_before = selector.dirichlets.unsqueeze(0).unsqueeze(0).expand(1, 1, H, C, C)
    alpha_cc_before, beta_cc_before = dirichlet_to_beta(dirichlets_before) # (1, 1, H, C)
    alpha_cc_before = alpha_cc_before.permute(0,3,1,2)  # (1, C, 1, H)
    beta_cc_before  = beta_cc_before.permute(0,3,1,2)   # (1, C, 1, H)
    pbest_rows_before = compute_p_best_beta_batched(alpha_cc_before, beta_cc_before).squeeze(-2) # (1, C, H)

    mixture0 = (selector.pi_hat[:, None] * pbest_rows_before).sum(1)   # (1,H)
    H_before = -(mixture0.clamp_min(1e-12)
                          .mul(mixture0.clamp_min(1e-12).log2())
                          ).sum(-1)

    # broadcast helpers
    mixture0_bc = mixture0.view(1, 1, H) # (1,1,H)
    pi_hat_row  = selector.pi_hat.view(1, C, 1)   # (1,C,1)

    eig_chunks = []
    for s in tqdm(range(0, len(candidates), chunk_size)):
        ids   = candidates[s:s + chunk_size] # (B,)
        preds = classifier_preds[ids].argmax(-1) # (B, H)
        pi_hat_xi = selector.pi_hat_xi[ids]

        # do all hypothetical updates at once
        alpha_reduced, beta_reduced = batch_update_beta(selector, preds, update_weight) # (B,H,C_)
        alpha_reduced = alpha_reduced.permute(0,2,1).unsqueeze(-2)  # (B, C_, 1, H)
        beta_reduced = beta_reduced.permute(0,2,1).unsqueeze(-2)    # (B, C_, 1, H)

        pbest_hypothetical_rows = compute_p_best_beta_batched(alpha_reduced, 
                                                              beta_reduced, 
                                                              num_points=num_points).squeeze(-2) # (B, C_, H)

        deltas = pi_hat_row * (pbest_hypothetical_rows - pbest_rows_before) # (B,C,H)
        mix_new = mixture0_bc + deltas # (B,C,H)
        H_after = -(mix_new.clamp_min(1e-12).mul(mix_new.clamp_min(1e-12).log2())).sum(-1) # (B,C)
        eig = H_before - (pi_hat_xi * H_after).sum(-1)      # (B,)
        eig_chunks.append(eig)

    return torch.cat(eig_chunks)

class CODA(ModelSelector):
    def __init__(self, dataset,
                 q='eig',
                 prefilter_fn='disagreement',
                 prefilter_n=0,
                 epsilon=0.0,
                 base_prior="diag",
                 alpha=0.9,
                 learning_rate=0.01
                 ):
        self.dataset = dataset
        self.device = dataset.preds.device
        self.H, self.N, self.C = dataset.preds.shape
        self.q = q
        self.prefilter_fn = prefilter_fn
        self.prefilter_n = prefilter_n
        self.epsilon = epsilon
        self.base_prior = base_prior

        # hyperparams
        self.base_strength = alpha
        self.prior_strength = (1 - alpha)
        self.update_strength = learning_rate

        # initialise dirichlets
        ens_pred = Ensemble(dataset.preds).get_preds()
        ens_pred_hard = ens_pred.argmax(-1)  # pseudo labels
        soft_conf = create_confusion_matrices(ens_pred_hard, dataset.preds, mode='soft')
        self.dirichlets = initialize_dirichlets(soft_conf,
                                                self.prior_strength,
                                                base_strength=self.base_strength,
                                                base_prior=self.base_prior)
        # class marginal distribution
        # self.pi_hat = update_pi_hat(self.dirichlets, dataset.preds)
        # self.pi_hat = self.pi_hat / self.pi_hat.sum()
        self.update_pi_hat()

        self.labeled_idxs, self.labels = [], []
        self.unlabeled_idxs = list(range(self.N))
        self.q_vals = []
        self.stochastic = False
        self.step = 0

    @classmethod
    def from_args(cls, dataset, args):
        return cls(dataset,
                   q=args.q,
                   prefilter_fn=args.prefilter_fn,
                   prefilter_n=args.prefilter_n,
                   epsilon=args.epsilon,
                   alpha=args.alpha,
                   learning_rate=args.learning_rate,
                   base_prior=args.base_prior)

    def _prefilter(self, idxs):
        if self.prefilter_fn == 'disagreement':
            maj, _ = torch.mode(self.dataset.preds.argmax(-1), dim=0)
            mask = (self.dataset.preds.argmax(-1) != maj).sum(0) > 0
            idxs = [i for i in idxs if mask[i]]
        if self.prefilter_n and len(idxs) > self.prefilter_n:
            idxs = random.sample(idxs, self.prefilter_n)
            self.stochastic = True
        return idxs

    def update_pi_hat(self):
        adjusted = torch.einsum('hcs, hns -> hnc', self.dirichlets, self.dataset.preds)
        # per item
        self.pi_hat_xi = adjusted.sum(0)
        self.pi_hat_xi = self.pi_hat_xi / self.pi_hat_xi.sum(dim=-1, keepdim=True).clamp_(min=1e-12)
        # marginal (entire dataset)
        self.pi_hat = self.pi_hat_xi.sum(0)
        self.pi_hat = self.pi_hat / self.pi_hat.sum()

    def get_next_item_to_label(self, step=None):
        cand = self._prefilter(self.unlabeled_idxs) or self.unlabeled_idxs
        if self.q == 'iid' or (self.epsilon and random.random() < self.epsilon):
            self.stochastic = True
            return random.choice(cand), 1 / len(cand)

        if self.q == 'eig':
            q_vals = eig_dirichlet_batched(self,
                                        #    self.dirichlets,
                                        #    self.dataset.preds.permute(1, 0, 2),
                                        #    self.pi_hat,
                                        #    self.pi_hat_xi,
                                           cand,
                                        #    chunk_size=8
                                           )
        else:
            raise NotImplementedError(self.q)

        if step is not None and _DEBUG_VIZ:
            mlflow.log_image(plot_bar(q_vals), key="EIG", step=step)

        best = q_vals.max()
        ties = torch.isclose(q_vals, best, rtol=1e-8)
        idx_local = random.choice(torch.nonzero(ties, as_tuple=True)[0].tolist()) \
                    if ties.sum() > 1 else torch.argmax(q_vals).item()
        if ties.sum() > 1:
            self.stochastic = True
        return cand[idx_local], q_vals[idx_local].item()

    def add_label(self, idx, true_class, selection_prob):
        preds = F.one_hot(self.dataset.preds[:, idx].argmax(-1), self.C).float()
        self.dirichlets[:, true_class] += self.update_strength * preds
        # self.pi_hat = update_pi_hat(self.dirichlets, self.dataset.preds)
        # self.pi_hat = self.pi_hat / self.pi_hat.sum()

        self.update_pi_hat()
        self.labeled_idxs.append(idx)
        self.labels.append(int(true_class))
        self.q_vals.append(selection_prob)
        self.unlabeled_idxs.remove(idx)

    def get_best_model_prediction(self):
        H, C, _ = self.dirichlets.shape
        expanded = self.dirichlets.unsqueeze(0).unsqueeze(0).expand(1, 1, H, C, C)
        p_best = p_best_row_mixture_batched(expanded, self.pi_hat).squeeze(0) # (H,)
        
        if torch.isnan(p_best).any():
            raise ValueError("NaN in posterior")
        
        if _DEBUG_VIZ: mlflow.log_image(plot_bar(p_best), key="PBest", step=self.step)

        # track how many times we've done this
        self.step += 1 

        return torch.argmax(p_best)
