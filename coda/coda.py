import torch
from torch.distributions import Beta
import torch.nn.functional as F
from tqdm import tqdm
import random
import mlflow

from coda.base import ModelSelector
from .surrogates import Ensemble

_DEBUG = True
from .logging_util import plot_bar


def distribution_entropy(prob: torch.Tensor, eps=1e-12) -> torch.Tensor:
    prob_clamped = prob.clamp(min=eps)
    return -(prob_clamped * prob_clamped.log2()).sum()


def _check(t: torch.Tensor, name: str, *, raise_err=True):
    # for debugging - check t for infs/nans
    if _DEBUG:
        bad = ~torch.isfinite(t)
        if bad.any():
            msg = (f"[NUMERIC ERROR] {name} has {bad.sum()} bad values "
                f"(NaN/Inf) out of {t.numel()} "
                f"min={t.min().item():.3g}, max={t.max().item():.3g}")
            if raise_err:
                raise RuntimeError(msg)
            print(msg)
    return t


def _check_prob(p: torch.Tensor, name="prob", eps=1e-12):
    # check if p is a valid probability distribution
    if _DEBUG:
        _check(p, name)
        if (p < -eps).any():
            raise RuntimeError(f"{name} has negatives")
        s = p.sum(-1)
        if (torch.isnan(s) | torch.isinf(s)).any():
            raise RuntimeError(f"{name} sum is nan/inf")
        if ((s - 1).abs() > 1e-4).any():
            print(f"[WARN] {name} rows not normalised: min sum={s.min():.4f}, "
                f"max sum={s.max():.4f}")


def dirichlet_to_beta(alpha_dirichlet: torch.Tensor):
    """
    Get parameters for beta distributions representing the diagonal.

    Args:
        alpha_dirichlet: shape TODO
    Returns:
        alpha_cc, beta_cc: shape TODO
    """
    H, C, _ = alpha_dirichlet.shape
    alpha_cc = alpha_dirichlet[:, torch.arange(C), torch.arange(C)]
    beta_cc  = alpha_dirichlet.sum(dim=2) - alpha_cc
    return alpha_cc, beta_cc


def create_confusion_matrices(true_labels: torch.Tensor,
                              model_predictions: torch.Tensor) -> torch.Tensor:
    H, N, C = model_predictions.shape
    dev = model_predictions.device
    true_one_hot = F.one_hot(true_labels, C).float().to(dev)
    preds = F.one_hot(model_predictions.argmax(-1), C).float()
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
            base = torch.full((C, C), base_strength / (C - 1),
                              dtype=soft_confusion.dtype,
                              device=soft_confusion.device)
            base.fill_diagonal_(base_strength)
    elif base_prior == "uniform":
        base = torch.full((C, C), 2 * base_strength / C,
                          dtype=soft_confusion.dtype,
                          device=soft_confusion.device)
    elif base_prior == "empirical-bayes":
        base = 2 * soft_confusion.mean(0) + 1e-5
    base = base.unsqueeze(0).expand(H, C, C)
    return base + prior_strength * soft_confusion


def compute_ensemble_marginal(confusion_matrices: torch.Tensor,
                              model_predictions: torch.Tensor) -> torch.Tensor:
    adjusted = torch.einsum('hni, hci -> hnc', model_predictions, confusion_matrices)
    marg = adjusted.sum((0, 1))
    return marg / (confusion_matrices.shape[0] * model_predictions.shape[1])


def batch_update_dirichlet_for_item(dirichlet_alphas: torch.Tensor,
                                    worker_preds: torch.Tensor,
                                    update_weight: float = 1.0) -> torch.Tensor:
    N, H, C = worker_preds.shape
    updated = dirichlet_alphas[None, None].expand(N, C, H, C, C).clone()
    updates = worker_preds[:, None].expand(-1, C, -1, -1) * update_weight
    for c in range(C):
        updated[:, c, :, c, :] += updates[:, c, :, :]
    return updated


def compute_p_best_beta_batched(alpha_batch: torch.Tensor,
                                beta_batch:  torch.Tensor,
                                num_points: int = 1024,
                                eps: float = 1e-30,
                                chunk_size: int = None) -> torch.Tensor:
    device = alpha_batch.device
    if alpha_batch.ndim == 2:
        alpha_batch = alpha_batch[None]
        beta_batch = beta_batch[None]
        single_item = True
    else:
        single_item = False

    N, C, H = alpha_batch.shape
    chunk_size = chunk_size or N
    x = torch.linspace(1e-6, 1 - 1e-6, num_points,
                       device=device).unsqueeze(-1)            # P×1
    prob_out = torch.zeros(N, C, H, device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        a_flat = alpha_batch[start:end].reshape(-1, H)          # B*C × H
        b_flat = beta_batch[start:end].reshape(-1, H)
        logpdf = Beta(a_flat.reshape(-1), b_flat.reshape(-1)).log_prob(x)
        pdf = logpdf.exp().T.reshape(-1, H, num_points)         # B*C × H × P
        _check(pdf, "pdf")

        cdf = torch.zeros_like(pdf)
        for j in range(1, num_points):
            dx = x[j] - x[j-1]
            cdf[:, :, j] = cdf[:, :, j-1] + 0.5*(pdf[:, :, j] + pdf[:, :, j-1])*dx
        _check(cdf, "cdf")

        log_cdf = torch.log(cdf.clamp_min(eps))
        prod_excl = torch.exp(log_cdf.sum(1, keepdim=True) - log_cdf)
        integrand = pdf * prod_excl
        _check(integrand, "integrand")

        prob = torch.trapz(integrand, x.squeeze(), dim=2)
        _check(prob, "Pbest(beta)") 
        prob = prob / prob.sum(-1, keepdim=True).clamp_min(eps)
        _check_prob(prob, "Pbest(beta)-norm")
        prob_out[start:end] = prob.reshape(end-start, C, H)

    return prob_out[0] if single_item else prob_out


def _p_best_row_mixture_batched(updated_dirichlet: torch.Tensor,
                                pi: torch.Tensor,
                                num_points: int = 1024) -> torch.Tensor:
    """
    updated_dirichlet : (B, C, H, C, C)
    pi                : (C,)

    Returns
    -------
    prob_best  : (B, C, H)  --  P(best | item b, true class = c)
    """

    B, C, H = updated_dirichlet.shape[:3]
    dev = updated_dirichlet.device

    # ----  α_cc  ------------------------------------------------------
    # diag  (B, C, H, C)
    diag_full = torch.diagonal(updated_dirichlet, dim1=-2, dim2=-1)
    idx = torch.arange(C, device=dev).view(1, C, 1, 1)               # (1,C,1,1)
    alpha_cc = torch.take_along_dim(diag_full, idx, dim=-1).squeeze(-1)   # (B,C,H)

    # ----  β_cc -------------------------------------------------------
    row_sum  = updated_dirichlet.sum(-1)                              # (B,C,H,C)
    beta_cc  = (torch.take_along_dim(row_sum, idx, dim=-1)
                .squeeze(-1) - alpha_cc)                              # (B,C,H)

    # ----  integrate Beta PDFs  --------------------------------------
    # compute_p_best_beta_batched expects (N,C,H); here N == B
    prob_best_bch = compute_p_best_beta_batched(alpha_cc,
                                                beta_cc,
                                                num_points=num_points)  # (B,C,H)
    return prob_best_bch


def eig_dirichlet_batched(dirichlet_alphas: torch.Tensor,
                          worker_preds: torch.Tensor,
                          marginal_distribution: torch.Tensor,
                          candidate_ids: list[int],
                          chunk_size: int = 100,
                          update_weight: float = 1.0,
                          num_points: int = 1024) -> torch.Tensor:

    device   = dirichlet_alphas.device
    _, H, C  = worker_preds.shape
    pi       = marginal_distribution

    candidates = torch.tensor(candidate_ids, device=device)

    # -------- current posterior entropy --------------------------------
    H, C_dir, _ = dirichlet_alphas.shape
    expanded = dirichlet_alphas.unsqueeze(0).unsqueeze(0).expand(1, C_dir, H, C_dir, C_dir)
    current_probs_c = _p_best_row_mixture_batched(expanded, pi,
                                                 num_points=num_points)  # (1,C,H)
    current_probs = (current_probs_c * pi.view(1, C_dir, 1)).sum(1).squeeze(0)
    H_current = distribution_entropy(current_probs)

    eig_chunks = []
    for s in tqdm(range(0, len(candidates), chunk_size)):
        ids   = candidates[s:s + chunk_size]              # (B,)
        preds = F.one_hot(worker_preds[ids].argmax(-1), C).float()  # (B,H,C)

        # (B,C,H,C,C)
        updated = batch_update_dirichlet_for_item(dirichlet_alphas,
                                                  preds,
                                                  update_weight)

        # --------------------------------------------------------------
        #   Compute posterior probabilities using the row-mixture method
        # --------------------------------------------------------------
        updated_probs = _p_best_row_mixture_batched(updated, pi,
                                                    num_points)   # (B,C,H)

        # entropy for each hypothetical class
        p_clamped = updated_probs.clamp_min(1e-12)
        H_updated = -(p_clamped * p_clamped.log2()).sum(-1)       # (B,C)

        # expected IG  = Σ_c π_c·(H_current − H_updated_c)
        eig_chunk = (H_current - H_updated) * pi
        eig_chunks.append(eig_chunk.sum(1))                       # (B,)

    return torch.cat(eig_chunks)                                  # (N_candidates,)


class CODA(ModelSelector):
    def __init__(self, dataset,
                 q='eig',
                 prefilter_fn='disagreement',
                 prefilter_n=0,
                 epsilon=0.0,
                 base_prior="diag",
                 temperature=1.0,
                 alpha=0.9,
                 learning_rate_ratio=0.01
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
        self.base_strength = alpha / temperature
        self.prior_strength = (1 - alpha) / temperature
        self.update_strength = learning_rate_ratio / temperature

        # initialise Dirichlets
        ens_pred = Ensemble(dataset.preds).get_preds().argmax(-1)
        soft_conf = create_confusion_matrices(ens_pred, dataset.preds, 'soft')
        self.dirichlets = initialize_dirichlets(soft_conf,
                                                self.prior_strength,
                                                base_strength=self.base_strength,
                                                base_prior=self.base_prior)

        self.pi_hat = compute_ensemble_marginal(self.dirichlets, dataset.preds)
        self.pi_hat = self.pi_hat / self.pi_hat.sum()

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
                   temperature=args.temperature,
                   alpha=args.alpha,
                   learning_rate_ratio=args.learning_rate_ratio,
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

    def get_next_item_to_label(self, step=None):
        cand = self._prefilter(self.unlabeled_idxs) or self.unlabeled_idxs
        if self.q == 'iid' or (self.epsilon and random.random() < self.epsilon):
            self.stochastic = True
            return random.choice(cand), 1 / len(cand)

        if self.q == 'eig':
            q_vals = eig_dirichlet_batched(self.dirichlets,
                                           self.dataset.preds.permute(1, 0, 2),
                                           self.pi_hat,
                                           cand,
                                           chunk_size=8)
        else:
            raise NotImplementedError(self.q)

        if step is not None:
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
        self.pi_hat = compute_ensemble_marginal(self.dirichlets, self.dataset.preds)
        self.pi_hat = self.pi_hat / self.pi_hat.sum()

        self.labeled_idxs.append(idx)
        self.labels.append(int(true_class))
        self.q_vals.append(selection_prob)
        self.unlabeled_idxs.remove(idx)

    def get_best_model_prediction(self):
        H, C, _ = self.dirichlets.shape
        expanded = self.dirichlets.unsqueeze(0).unsqueeze(0).expand(1, C, H, C, C)
        probs_c = _p_best_row_mixture_batched(expanded,
                                              self.pi_hat)
        p_best = (probs_c * self.pi_hat.view(1, C, 1)).sum(1).squeeze(0)
        
        if torch.isnan(p_best).any():
            raise ValueError("NaN in posterior")
        
        mlflow.log_image(plot_bar(p_best), key="PBest", step=self.step)

        # track how many times we've done this
        self.step += 1 

        return torch.argmax(p_best)
