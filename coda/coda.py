import torch
from torch.distributions import Beta
import torch.nn.functional as F
from tqdm import tqdm
import random
import mlflow

from coda.base import ModelSelector
from surrogates import Ensemble

_DEBUG = True
from logging_util import plot_bar


# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------

def distribution_entropy(prob: torch.Tensor, eps=1e-12) -> torch.Tensor:
    prob_clamped = prob.clamp(min=eps)
    return -(prob_clamped * prob_clamped.log2()).sum()

def _check(t: torch.Tensor, name: str, *, raise_err=True):
    """Assert tensor has no NaN/±Inf and print useful stats."""
    if _DEBUG:
        bad = ~torch.isfinite(t)
        if bad.any():
            msg = (f"[NUMERIC ERROR] {name} has {bad.sum()} bad values "
                f"(NaN/Inf) out of {t.numel()} "
                f"min={t.min().item():.3g}, max={t.max().item():.3g}")
            if raise_err:
                raise RuntimeError(msg)
            print(msg)
    return t  # so you can inline it

def _check_prob(p: torch.Tensor, name="prob", eps=1e-12):
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

# ---------------------------------------------------------------------
# Dirichlet → Beta helpers
# ---------------------------------------------------------------------

def dirichlet_to_beta(alpha_dirichlet: torch.Tensor):
    """Row‑wise conversion: returns α_cc, β_cc  (H,C)."""
    H, C, _ = alpha_dirichlet.shape
    alpha_cc = alpha_dirichlet[:, torch.arange(C), torch.arange(C)]
    beta_cc  = alpha_dirichlet.sum(dim=2) - alpha_cc
    return alpha_cc, beta_cc


def dirichlet_to_beta_with_marginal(dirichlet_alphas: torch.Tensor,
                                    marginal_distribution: torch.Tensor):
    """
    Aggregated pseudo‑counts (“counts” approximation).
    Returns (H,) α,β.
    """
    diag_counts = dirichlet_alphas.diagonal(dim1=-2, dim2=-1)       # (...,C)
    row_sums    = dirichlet_alphas.sum(dim=-1)                      # (...,C)
    alpha_beta  = (marginal_distribution * diag_counts).sum(dim=-1)
    beta_beta   = (marginal_distribution * (row_sums - diag_counts)).sum(dim=-1)
    return alpha_beta.clamp(min=1e-3), beta_beta.clamp(min=1e-3)


def dirichlet_to_beta_overall(dirichlet_alphas: torch.Tensor,
                              marginal_distribution: torch.Tensor,
                              approx: str = "counts"):
    """
    Shared helper: given ...×C×C Dirichlet, return α,β per leading entry
    according to 'counts', 'mom', or 'inflated'.
    """
    if approx == "counts":
        return dirichlet_to_beta_with_marginal(dirichlet_alphas,
                                               marginal_distribution)

    # Common pieces for MoM and inflated
    C = dirichlet_alphas.shape[-1]
    diag = dirichlet_alphas[..., torch.arange(C), torch.arange(C)]      # ...×C
    row_sum = dirichlet_alphas.sum(dim=-1)                              # ...×C
    row_mean = diag / row_sum                                           # ...×C
    row_var  = (diag * (row_sum - diag)) / (row_sum**2 * (row_sum + 1)) # ...×C

    mu = (marginal_distribution * row_mean).sum(dim=-1)                 # ...
    var_within = (marginal_distribution**2 * row_var).sum(dim=-1)       # ...

    if approx == "mom":
        total_var = var_within
    elif approx == "inflated":
        between = (marginal_distribution *
                   (row_mean - mu.unsqueeze(-1))**2).sum(dim=-1)
        total_var = var_within + between
    else:
        raise ValueError(f"Unknown beta approximation: {approx}")

    eps = 1e-6
    mu_safe = mu.clamp(eps, 1 - eps)
    var_max = mu_safe * (1 - mu_safe)
    var_clamped = torch.min(total_var, var_max - eps)

    nu = mu_safe * (1 - mu_safe) / (var_clamped + eps) - 1
    nu = nu.clamp(min=eps)

    alpha_beta = mu_safe * nu
    beta_beta  = (1 - mu_safe) * nu
    return alpha_beta, beta_beta


# ---------------------------------------------------------------------
# Single‑Beta “probability best” for arbitrary α,β
# ---------------------------------------------------------------------

def compute_p_best_beta(alpha: torch.Tensor,
                        beta: torch.Tensor,
                        num_points: int = 1024,
                        eps: float = 1e-30) -> torch.Tensor:
    device = alpha.device
    x = torch.linspace(1e-6, 1 - 1e-6, num_points, device=device)     # (P,)
    pdf = torch.exp(Beta(alpha, beta).log_prob(x.unsqueeze(1))).T     # (H,P)
    _check(pdf, "pdf")

    # cdf = torch.zeros_like(pdf)
    # for j in range(1, num_points):
    #     dx = x[j] - x[j - 1]
    #     cdf[:, j] = cdf[:, j - 1] + 0.5 * (pdf[:, j] + pdf[:, j - 1]) * dx
    dx   = x[1] - x[0]                                              # scalar
    cdf  = torch.cumsum(pdf, dim=-1) - 0.5*pdf[:, 0:1] - 0.5*pdf    # rectangle rule
    cdf  = cdf * dx                                                 # now exact trapezoid integral

    log_cdf = torch.log(cdf.clamp_min(eps))
    prod_excl = torch.exp(log_cdf.sum(0).unsqueeze(0) - log_cdf)
    prob_best = torch.trapz(pdf * prod_excl, x, dim=1)
    return prob_best / prob_best.sum()

# ---------------------------------------------------------------------
# Dirichlet → P(best) with selectable approximation
# ---------------------------------------------------------------------

def compute_p_best_dirichlet(alpha_dirichlet: torch.Tensor,
                             marginal_distribution: torch.Tensor,
                             num_points: int = 1024,
                             approx=None) -> torch.Tensor:
    print("approx", approx, type(approx))
    if approx is None or approx.lower() == "none":
        print("Computing full PBest")
        return compute_p_best_row_mixture(alpha_dirichlet, marginal_distribution, num_points=num_points)
    elif approx == "mom":
        alpha_cc, beta_cc = dirichlet_to_beta(alpha_dirichlet)
        return compute_p_best_mom(alpha_cc, beta_cc, marginal_distribution, num_points)
    else:
        # 'counts' or 'inflated'
        a, b = dirichlet_to_beta_overall(alpha_dirichlet, marginal_distribution, approx)
        return compute_p_best_beta(a, b, num_points)

# ---------------------------------------------------------------------
# Method of moments beta approx
# ---------------------------------------------------------------------

def compute_p_best_mom(alpha: torch.Tensor,
                       beta: torch.Tensor,
                       marginal_distribution: torch.Tensor,
                       num_points: int = 1024) -> torch.Tensor:
    mean = (marginal_distribution * (alpha / (alpha + beta))).sum(1)
    var = (marginal_distribution**2 *
           (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))).sum(1)
    eps = 1e-6
    mu = mean.clamp(eps, 1 - eps)
    var_clamped = torch.min(var, mu * (1 - mu) - eps)
    nu = mu * (1 - mu) / (var_clamped + eps) - 1
    nu = nu.clamp(min=eps)
    a, b = mu * nu, (1 - mu) * nu
    return compute_p_best_beta(a, b, num_points)

# ---------------------------------------------------------------------
# Confusion‑matrix helpers (unchanged)
# ---------------------------------------------------------------------

def create_confusion_matrices(true_labels: torch.Tensor,
                              model_predictions: torch.Tensor,
                              mode: str = 'hard') -> torch.Tensor:
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

# ---------------------------------------------------------------------
# Batched Dirichlet update helper
# ---------------------------------------------------------------------

def batch_update_dirichlet_for_item(dirichlet_alphas: torch.Tensor,
                                    worker_preds: torch.Tensor,
                                    update_weight: float = 1.0) -> torch.Tensor:
    N, H, C = worker_preds.shape
    updated = dirichlet_alphas[None, None].expand(N, C, H, C, C).clone()
    updates = worker_preds[:, None].expand(-1, C, -1, -1) * update_weight
    for c in range(C):
        updated[:, c, :, c, :] += updates[:, c, :, :]
    return updated

# ---------------------------------------------------------------------
# Beta integration on batches 
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Row-wise mixture:  P(best) = Σ_c π_c · P(best | class = c)
# ---------------------------------------------------------------------
def compute_p_best_row_mixture(alpha_dirichlet: torch.Tensor,
                               marginal_distribution: torch.Tensor,
                               num_points: int = 1024) -> torch.Tensor:
    """
    No collapse to a single Beta.
    For each class c:
        • turn the Dirichlet row into Beta(α_cc, β_cc)  (per model)
        • integrate P(best) across models
    Weight those per-class results by π_c and renormalise.

    Args
    ----
    alpha_dirichlet : (H, C, C)  posterior Dirichlet rows
    marginal_distribution : (C,)  class priors  π
    num_points : integration grid size for Beta PDFs

    Returns
    -------
    prob_best : (H,)  probability each model is best
    """

    # 1.  Get per-class Beta parameters  (H, C)
    alpha_cc, beta_cc = dirichlet_to_beta(alpha_dirichlet)

    H, C = alpha_cc.shape
    device = alpha_cc.device
    prob_best = torch.zeros(H, device=device)

    # 2.  Loop over classes (C is usually ≪ H, so this is cheap)
    for c in range(C):
        p_best_c = compute_p_best_beta(alpha_cc[:, c],
                                       beta_cc[:, c],
                                       num_points=num_points)   # (H,)
        _check(p_best_c, f"p_best_c={c}")
        prob_best += marginal_distribution[c] * p_best_c        # weight by π_c

    # 3.  Safeguard & normalise
    prob_best = prob_best / prob_best.sum().clamp_min(1e-12)
    _check_prob(prob_best, "prob_best")
    return prob_best

# ---------------------------------------------------------------------
# fast row-mixture helper  (vectorised, no Python loops)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Expected-Information-Gain with selectable Beta / mixture posterior
# ---------------------------------------------------------------------

def eig_dirichlet_batched(dirichlet_alphas: torch.Tensor,
                          worker_preds: torch.Tensor,
                          marginal_distribution: torch.Tensor,
                          candidate_ids: list[int],
                          chunk_size: int = 100,
                          update_weight: float = 1.0,
                          update_rule: str = "hard",
                          beta_approx=None,
                          num_points: int = 1024) -> torch.Tensor:

    device   = dirichlet_alphas.device
    _, H, C  = worker_preds.shape
    pi       = marginal_distribution

    candidates = torch.tensor(candidate_ids, device=device)

    # -------- current posterior entropy --------------------------------
    current_probs = compute_p_best_dirichlet(dirichlet_alphas, pi,
                                             approx=beta_approx, # TODO
                                             num_points=num_points)
    H_current = distribution_entropy(current_probs)

    eig_chunks = []
    for s in tqdm(range(0, len(candidates), chunk_size)):
        ids   = candidates[s:s + chunk_size]              # (B,)
        preds = worker_preds[ids]                         # (B,H,C)

        if update_rule == "hard":
            preds = F.one_hot(preds.argmax(-1), C).float()

        # (B,C,H,C,C)
        updated = batch_update_dirichlet_for_item(dirichlet_alphas,
                                                  preds,
                                                  update_weight)

        # --------------------------------------------------------------
        #   Choose posterior style
        # --------------------------------------------------------------
        if beta_approx is None:
            # exact mixture, now vectorised
            updated_probs = _p_best_row_mixture_batched(updated, pi,
                                                        num_points)   # (B,C,H)
        else:
            # existing one-Beta approximations
            flat = updated.reshape(-1, C, C)                     # (B*C*H, C, C)
            a_flat, b_flat = dirichlet_to_beta_overall(flat, pi, approx=beta_approx)
            a_batch = a_flat.reshape(len(ids), C, H)
            b_batch = b_flat.reshape(len(ids), C, H)
            updated_probs = compute_p_best_beta_batched(a_batch, b_batch,
                                                        num_points=num_points)

        # entropy for each hypothetical class
        p_clamped = updated_probs.clamp_min(1e-12)
        H_updated = -(p_clamped * p_clamped.log2()).sum(-1)       # (B,C)

        # expected IG  = Σ_c π_c·(H_current − H_updated_c)
        eig_chunk = (H_current - H_updated) * pi
        eig_chunks.append(eig_chunk.sum(1))                       # (B,)

    return torch.cat(eig_chunks)                                  # (N_candidates,)


# ---------------------------------------------------------------------
# CODA model‑selector class
# ---------------------------------------------------------------------

class CODA(ModelSelector):
    def __init__(self, dataset,
                 prior_source="ens",
                 q='eig',
                 prefilter_fn='disagreement',
                 prefilter_n=0,
                 epsilon=0.0,
                 update_rule="hard",
                 base_prior="diag",
                 temperature=1.0,
                 alpha=0.9,
                 learning_rate_ratio=0.01,
                 beta_approx=None
                 ):
        self.dataset = dataset
        self.device = dataset.preds.device
        self.H, self.N, self.C = dataset.preds.shape
        self.q = q
        self.prefilter_fn = prefilter_fn
        self.prefilter_n = prefilter_n
        self.epsilon = epsilon
        self.update_rule = update_rule
        self.base_prior = base_prior
        self.beta_approx = beta_approx

        # hyper‐params → strengths
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

    # ---------------------

    @classmethod
    def from_args(cls, dataset, args):
        return cls(dataset,
                   prior_source=args.prior_source,
                   q=args.q,
                   prefilter_fn=args.prefilter_fn,
                   prefilter_n=args.prefilter_n,
                   epsilon=args.epsilon,
                   update_rule=args.update_rule,
                   temperature=args.temperature,
                   alpha=args.alpha,
                   learning_rate_ratio=args.learning_rate_ratio,
                   base_prior=args.base_prior,
                   beta_approx=args.beta_approx)

    # ---------------------

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
                                           chunk_size=8,
                                           beta_approx=self.beta_approx)
        else:
            raise NotImplementedError(self.q)

        if step is not None:
            print("Lggig EIG")
            mlflow.log_image(plot_bar(q_vals), key="EIG", step=step)
        else:
            print("NOT logging eig")

        best = q_vals.max()
        ties = torch.isclose(q_vals, best, rtol=1e-8)
        idx_local = random.choice(torch.nonzero(ties, as_tuple=True)[0].tolist()) \
                    if ties.sum() > 1 else torch.argmax(q_vals).item()
        if ties.sum() > 1:
            self.stochastic = True
        return cand[idx_local], q_vals[idx_local].item()

    def add_label(self, idx, true_class, selection_prob):
        preds = self.dataset.preds[:, idx]
        if self.update_rule == "hard":
            preds = F.one_hot(preds.argmax(-1), self.C).float()
        self.dirichlets[:, true_class] += self.update_strength * preds
        self.pi_hat = compute_ensemble_marginal(self.dirichlets, self.dataset.preds)
        self.pi_hat = self.pi_hat / self.pi_hat.sum()

        self.labeled_idxs.append(idx)
        self.labels.append(int(true_class))
        self.q_vals.append(selection_prob)
        self.unlabeled_idxs.remove(idx)

    def get_best_model_prediction(self, step=None):
        p_best = compute_p_best_dirichlet(self.dirichlets,
                                          self.pi_hat,
                                          approx=None ) #self.beta_approx)
        
        if torch.isnan(p_best).any():
            raise ValueError("NaN in posterior")
        
        if step is not None:
            mlflow.log_image(plot_bar(p_best), key="PBest", step=step)

        return torch.argmax(p_best)