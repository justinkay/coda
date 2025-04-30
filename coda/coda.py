import torch
from torch.distributions import Beta
import torch.nn.functional as F
from tqdm import tqdm
import random

from coda.base import ModelSelector
from coda.beta import distribution_entropy, sample_is_best_worker_beta_batched
from surrogates import Ensemble


def dirichlet_to_beta(alpha_dirichlet: torch.Tensor):
    """
    Converts Dirichlet parameters for a confusion matrix row (true class c)
    into Beta parameters for the diagonal element (accuracy on class c).
    
    Args:
        alpha_dirichlet: Shape (H, C, C) where alpha_dirichlet[h, c, :] 
                         are Dirichlet parameters for model h, true class c.
    
    Returns:
        alpha_beta: Shape (H, C) of Beta alpha parameters for each (h, c).
        beta_beta: Shape (H, C) of Beta beta parameters for each (h, c).
    """
    H, C, _ = alpha_dirichlet.shape
    alpha_cc = alpha_dirichlet[:, torch.arange(C), torch.arange(C)]  # (H, C)
    beta_cc = alpha_dirichlet.sum(dim=2) - alpha_cc  # Sum over k≠c: (H, C)
    return alpha_cc, beta_cc

def sample_is_best_worker_class_conditional(
    alpha: torch.Tensor,  # shape (H, C)
    beta: torch.Tensor,    # shape (H, C)
    marginal_distribution: torch.Tensor,  # shape (C,)
    num_points=128
) -> torch.Tensor:
    """
    Computes the probability that each model is the best, given class-conditional beta distributions
    and marginal class probabilities. Uses method of moments to approximate each model's overall
    accuracy distribution with a Beta distribution.
    
    Args:
        alpha: Tensor of shape (H, C) containing alpha parameters for H models and C classes.
        beta: Tensor of shape (H, C) containing beta parameters for H models and C classes.
        marginal_distribution: Tensor of shape (C,) with marginal probabilities for each class.
        num_points: Number of grid points for numerical integration.
        
    Returns:
        prob_best: Tensor of shape (H,) with probabilities that each model is the best.
    """
    device = alpha.device
    H, C = alpha.shape

    # Compute mean and variance for each model's overall accuracy
    mean = torch.sum(marginal_distribution * (alpha / (alpha + beta)), dim=1)  # (H,)
    var = torch.sum((marginal_distribution**2) * (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)), dim=1)  # (H,)

    # Fit Beta distribution using method of moments
    mu = mean
    v = var

    epsilon = 1e-6
    safe_mu = mu.clamp(min=epsilon, max=1 - epsilon)
    variance_bound = safe_mu * (1 - safe_mu)
    v_clamped = torch.min(v, variance_bound - epsilon)

    nu = (safe_mu * (1 - safe_mu) / (v_clamped + epsilon)) - 1
    nu = torch.clamp(nu, min=epsilon)

    alpha_prime = safe_mu * nu
    beta_prime = (1 - safe_mu) * nu

    # Ensure valid Beta parameters
    alpha_prime = torch.clamp(alpha_prime, min=epsilon)
    beta_prime = torch.clamp(beta_prime, min=epsilon)

    # Use existing grid-based integration with approximated Beta parameters
    return sample_is_best_worker(alpha_prime, beta_prime, num_points)


def compute_pdf_for_beta_grid(alpha: torch.Tensor, beta: torch.Tensor, num_points=128):
    """
    Given alpha, beta of shape (H, ), compute a grid-based PDF for each h in [0..H-1].
    Returns:
      x: (num_points,) the grid in [0,1]
      pdf_vals: (H, num_points) the PDF for each model h at each x-grid point
    """
    device = alpha.device
    H = alpha.shape[0]

    eps = 1e-6
    x = torch.linspace(eps, 1-eps, num_points, device=device)  # (num_points,)

    # PDF for Beta(alpha, beta) at each x
    # We'll replicate your single-Beta approach:
    dist_betas = torch.distributions.Beta(alpha, beta)  # shape (H,) "batch" of Beta
    # Evaluate PDF at shape (num_points, H), then transpose => (H, num_points)
    pdf_vals = torch.exp(dist_betas.log_prob(x.unsqueeze(1))).transpose(0, 1).contiguous()

    return x, pdf_vals

def trapezoid_cdf_from_pdf(pdf_vals: torch.Tensor, x: torch.Tensor):
    """
    Given pdf_vals of shape (H, num_points), integrate from 0..x_j to get the CDF.
    We'll do a cumulative trapezoidal integration along the x-dimension.
    Returns cdf_vals of the same shape (H, num_points).
    """
    device = pdf_vals.device
    H, num_points = pdf_vals.shape
    cdf_vals = torch.zeros_like(pdf_vals)

    # For j in [1..num_points-1], increment by the trapezoid area from x[j-1] to x[j].
    for j in range(1, num_points):
        dx = x[j] - x[j - 1]
        trap = 0.5 * (pdf_vals[:, j] + pdf_vals[:, j - 1]) * dx
        cdf_vals[:, j] = cdf_vals[:, j - 1] + trap

    return cdf_vals

def compute_mixture_pdf_cdf_dirichlet(
    alpha_per_class: torch.Tensor,  # (H, C)
    beta_per_class: torch.Tensor,   # (H, C)
    pi_c: torch.Tensor,             # (C,)
    num_points: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a mixture PDF/CDF for each model h by summing over c in [1..C]:
      mixture_pdf[h, x] = sum_c pi_c[c] * BetaPDF_{h,c}(x)
      mixture_cdf[h, x] = sum_c pi_c[c] * BetaCDF_{h,c}(x)

    Returns:
      x: (num_points,)
      mixture_pdf: (H, num_points)
      mixture_cdf: (H, num_points)
    """
    device = alpha_per_class.device
    H, C = alpha_per_class.shape

    # Prepare to accumulate mixture pdf/cdf
    mixture_pdf = None
    mixture_cdf = None
    x_final = None

    for c_idx in range(C):
        alpha_c = alpha_per_class[:, c_idx]  # shape (H,)
        beta_c  = beta_per_class[:, c_idx]   # shape (H,)

        # 1) build PDF for each h
        x, pdf_vals_c = compute_pdf_for_beta_grid(alpha_c, beta_c, num_points=num_points)
        # 2) build CDF from PDF
        cdf_vals_c = trapezoid_cdf_from_pdf(pdf_vals_c, x)

        # Weighted by pi_c[c_idx]
        w = pi_c[c_idx]
        weighted_pdf = w * pdf_vals_c  # (H, num_points)
        weighted_cdf = w * cdf_vals_c  # (H, num_points)

        if mixture_pdf is None:
            # initialize
            mixture_pdf = weighted_pdf
            mixture_cdf = weighted_cdf
            x_final = x
        else:
            # accumulate
            mixture_pdf = mixture_pdf + weighted_pdf
            mixture_cdf = mixture_cdf + weighted_cdf

    return x_final, mixture_pdf, mixture_cdf


# new integral method -- more principled; double checking it doesn't make things worse
# def sample_is_best_worker_class_conditional_new(
#     alpha_per_class: torch.Tensor,  # shape (H, C)
#     beta_per_class: torch.Tensor,   # shape (H, C)
#     pi_c: torch.Tensor,             # shape (C,)
#     num_points: int = 128,
#     eps: float = 1e-30
# ):
#     """
#     Drop-in replacement that does not rely on method-of-moments for overall accuracy.
#     Instead, each model's 'overall accuracy' distribution is a mixture across classes.

#     We compute mixture_pdf[h,x], mixture_cdf[h,x] for each model h. Then:
#       integrand[h,x] = mixture_pdf[h,x] * prod_{k != h} mixture_cdf[k,x]
#     We integrate over x via trapezoid rule, and normalize so sum=1.

#     Returns:
#       prob_best: (H,) probability that each model is best
#     """
#     device = alpha_per_class.device
#     H, C = alpha_per_class.shape

#     # 1) build mixture PDF and CDF for each model
#     x, mixture_pdf, mixture_cdf = compute_mixture_pdf_cdf_dirichlet(
#         alpha_per_class, beta_per_class, pi_c, num_points
#     )  # mixture_pdf,h => (H,num_points), mixture_cdf => (H,num_points)

#     # 2) Compute product_{k != h} mixture_cdf[k,x] in a stable way.
#     # We can do log-sum with sum_all, then subtract log for h => standard trick.
#     mixture_cdf_clamped = mixture_cdf.clamp(min=eps)
#     log_cdf = mixture_cdf_clamped.log()  # (H,num_points)
#     sum_all = log_cdf.sum(dim=0, keepdim=True)  # (1, num_points)
#     log_excl = sum_all - log_cdf  # (H, num_points)
#     product_excl = torch.exp(log_excl)  # (H, num_points)

#     # integrand[h,x] = mixture_pdf[h,x] * product_excl[h,x]
#     integrand = mixture_pdf * product_excl  # (H, num_points)

#     # 3) Integrate each row over x
#     prob_best = torch.trapz(integrand, x, dim=1)  # shape (H,)

#     # 4) normalize
#     sum_pb = prob_best.sum()
#     if sum_pb > 0:
#         prob_best = prob_best / sum_pb

#     return prob_best

def sample_is_best_worker(alpha: torch.Tensor, beta: torch.Tensor, num_points=128) -> torch.Tensor:
    """
    Original implementation for single Beta distribution per model.
    """
    device = alpha.device
    K = alpha.shape[0]

    eps = 1e-6
    x = torch.linspace(eps, 1.0-eps, num_points, device=device)
    dist_betas = torch.distributions.Beta(alpha, beta)
    pdf_vals = torch.exp(dist_betas.log_prob(x.unsqueeze(1))).transpose(0, 1)

    # Trapezoidal integration for CDF
    cdf_vals = torch.zeros_like(pdf_vals)
    for j in range(1, num_points):
        dx = x[j] - x[j - 1]
        trap = 0.5 * (pdf_vals[:, j] + pdf_vals[:, j - 1]) * dx
        cdf_vals[:, j] = cdf_vals[:, j - 1] + trap

    log_cdfs = torch.log(cdf_vals.clamp_min(1e-30))
    sum_log_cdfs = log_cdfs.sum(dim=0)

    log_exclusive = sum_log_cdfs.unsqueeze(0) - log_cdfs
    product_exclusive = torch.exp(log_exclusive)
    integrand = pdf_vals * product_exclusive

    prob_best = torch.trapz(integrand, x, dim=1)
    prob_best = prob_best / prob_best.sum()

    return prob_best

def sample_is_best_worker_dirichlet(
    alpha_dirichlet: torch.Tensor,  # Shape (H, C, C)
    marginal_distribution: torch.Tensor,  # Shape (C,)
    num_points=128
) -> torch.Tensor:
    """
    Computes the probability that each model is best when each row of the confusion matrix
    (for true class c) is modeled as a Dirichlet distribution.
    """
    # Extract Beta parameters for diagonal (accuracy) terms
    alpha_beta, beta_beta = dirichlet_to_beta(alpha_dirichlet)
    
    # Use the previously defined class-conditional method
    return sample_is_best_worker_class_conditional(
        alpha_beta, 
        beta_beta,
        marginal_distribution,
        num_points=num_points
    )

def create_confusion_matrices(true_labels: torch.Tensor, 
                             model_predictions: torch.Tensor, 
                             mode: str = 'hard') -> torch.Tensor:
    """
    Args:
        true_labels: (N,) - integer class labels
        model_predictions: (H, N, C) - predicted probabilities/logits
        mode: 'hard' (argmax) or 'soft' (probabilities)
    
    Returns:
        confusion_matrices: (H, C, C)
    """
    H, N, C = model_predictions.shape
    device = model_predictions.device
    
    # Convert true labels to one-hot (N, C)
    true_one_hot = F.one_hot(true_labels, num_classes=C).float().to(device)
    
    if mode == 'hard':
        # Get predicted class indices and convert to one-hot
        pred_classes = torch.argmax(model_predictions, dim=-1)  # (H, N)
        preds = F.one_hot(pred_classes, num_classes=C).float()  # (H, N, C)
    elif mode == 'soft':
        # Use probabilities directly
        preds = model_predictions  # (H, N, C)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    confusion = torch.einsum('nc, hnj -> hcj', true_one_hot, preds)  # (H, C, C)
    confusion_normalized = confusion / confusion.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    return confusion_normalized

def initialize_dirichlets_with_prior(soft_confusion: torch.Tensor, prior_strength: float, 
                                    base_strength=1.0, base_prior="diag", upweight_binary=False) -> torch.Tensor:
    """
    Initialize Dirichlet parameters for each model's confusion matrix.
    
    Args:
        soft_confusion: Tensor of shape (H, C, C) representing soft confusion matrices.
        prior_strength: Scaling factor for the soft confusion contribution.
    
    Returns:
        dirichlets: Tensor of shape (H, C, C) representing the Dirichlet alphas.
    """
    H, C, _ = soft_confusion.shape

    if base_prior == "diag":

        if C == 2 and upweight_binary:
            base = torch.full((C, C), base_strength*3.0/8.0, dtype=soft_confusion.dtype, device=soft_confusion.device)
            base.fill_diagonal_(base_strength*5.0/8.0)
        else:
            # Create a base confusion matrix for one model: 1.0 on the diagonal and 1.0/(C-1) off-diagonals.
            base = torch.full((C, C), base_strength / (C - 1), dtype=soft_confusion.dtype, device=soft_confusion.device)
            base.fill_diagonal_(base_strength)

    elif base_prior == "uniform":
        # also make sure every row sums to 2
        base = torch.full((C, C), 2 *base_strength / C, dtype=soft_confusion.dtype, device=soft_confusion.device)
    elif base_prior == "empirical-bayes":
        # also make sure every row sums to 2
        base = 2 * soft_confusion.mean(dim=0) + 1e-5

    # Expand the base matrix to all H models.
    base = base.unsqueeze(0).expand(H, C, C)

    # Add the scaled soft_confusion.
    dirichlets = base + prior_strength * soft_confusion
    return dirichlets

def compute_ensemble_marginal(
    confusion_matrices: torch.Tensor,  # Shape (H, C, C) for H models, C classes
    model_predictions: torch.Tensor,   # Shape (H, N, C) for N samples
) -> torch.Tensor:
    """
    Compute the ensemble marginal class distribution by combining:
    - Model predictions weighted by their confusion matrices
    - Aggregated over all samples and models
    """
    # For each model and sample, compute confusion-calibrated probabilities
    adjusted_predictions = torch.einsum(
        'hni, hci -> hnc', 
        model_predictions, 
        confusion_matrices
    )  # Shape (H, N, C)

    # Sum over all models and samples, then normalize
    marginal = adjusted_predictions.sum(dim=(0, 1))  # Sum over H and N
    marginal = marginal / (confusion_matrices.shape[0] * model_predictions.shape[1])  # Normalize

    return marginal



def dirichlet_to_beta_with_marginal(dirichlet_alphas, marginal_distribution):
    """
    Convert Dirichlet confusion matrices to Beta parameters for overall accuracy.
    Instead of moment matching, we aggregate pseudo-counts using the marginal
    class probabilities.
    
    Args:
        dirichlet_alphas: (H, C, C) - Dirichlet parameters for H models, one per row/class.
        marginal_distribution: (C,) - class prior probabilities (summing to 1)
        
    Returns:
        alpha_beta, beta_beta: (H,) Beta parameters for each model's overall accuracy.
    """
    # Get counts for correct predictions per class (H, C)
    diag_counts = dirichlet_alphas.diagonal(dim1=1, dim2=2)
    # Total counts per class (H, C)
    row_sums = dirichlet_alphas.sum(dim=-1)
    
    # Aggregate weighted counts across classes
    alpha_beta = (marginal_distribution * diag_counts).sum(dim=1)
    beta_beta = (marginal_distribution * (row_sums - diag_counts)).sum(dim=1)
    
    # Ensure nonzero parameters
    return alpha_beta.clamp(min=1e-3), beta_beta.clamp(min=1e-3)

# def batch_update_dirichlet_for_item_hard(dirichlet_alphas, worker_preds, C, update_weight=1.0):
#     """
#     Batched update of Dirichlet parameters for hypothetical class assignments
#     Args:
#         dirichlet_alphas: (H, C, C) - current Dirichlet parameters
#         worker_preds: (N, H) - worker predictions for items
#         C: number of classes
#     Returns:
#         updated_dirichlets: (N, C, H, C, C) updated parameters for each item-class hypothesis
#     """
#     H = dirichlet_alphas.shape[0]
#     N = worker_preds.shape[0]
    
#     # Create expanded base parameters (N, C, H, C, C)
#     updated = dirichlet_alphas.view(1, 1, H, C, C).expand(N, C, H, C, C).clone()
    
#     # # Create indices for updates
#     # true_classes = torch.arange(C, device=dirichlet_alphas.device).view(1, C, 1, 1, 1)
#     # pred_classes = worker_preds.view(N, 1, H, 1, 1)
    
#     # # Only update diagonal where true class == predicted class
#     # mask = (true_classes == pred_classes)
#     # updated[mask] += update_weight

#     N, H = worker_preds.shape[0], dirichlet_alphas.shape[0]
#     n_idx = torch.arange(N, device=dirichlet_alphas.device).view(N, 1, 1).expand(N, C, H)
#     c_idx = torch.arange(C, device=dirichlet_alphas.device).view(1, C, 1).expand(N, C, H)
#     h_idx = torch.arange(H, device=dirichlet_alphas.device).view(1, 1, H).expand(N, C, H)
#     pred_idx = worker_preds.unsqueeze(1).expand(N, C, H)  # The predicted class for each (n, h)

#     # Now update the specific cell (row=c, column=pred_idx) in each model's confusion matrix.
#     updated[n_idx, c_idx, h_idx, c_idx, pred_idx] += update_weight

#     return updated

# def eig_dirichlet_batched_hard(dirichlet_alphas, 
#                          worker_preds,
#                          marginal_distribution,
#                          candidate_ids,
#                          chunk_size=100):
#     """
#     Batched EIG computation using Dirichlet confusion matrices
#     Args:
#         dirichlet_alphas: (H, C, C) - current Dirichlet parameters
#         worker_preds: (N, H) - worker predictions for all items
#         marginal_distribution: (C,) - class prior probabilities
#         candidate_ids: List[int] - indices of candidate items to evaluate
#     Returns:
#         eig_values: (len(candidate_ids), EIG for each candidate item
#     """
#     device = dirichlet_alphas.device
#     candidates = torch.tensor(candidate_ids, device=device)
#     H, C, _ = dirichlet_alphas.shape
    
#     # Precompute current Beta approximation
#     # alpha_beta, beta_beta = dirichlet_to_beta_with_marginal(dirichlet_alphas, marginal_distribution)
#     # current_probs = sample_is_best_worker(alpha_beta, beta_beta)
#     current_probs = sample_is_best_worker_dirichlet(dirichlets, ensemble_marginal)
#     H_current = distribution_entropy(current_probs)
    
#     # Batch process candidates
#     eig_results = []
#     for chunk_start in range(0, len(candidates), chunk_size):
#         chunk = candidates[chunk_start:chunk_start+chunk_size]
#         chunk_preds = worker_preds[chunk]
        
#         # Get updated Dirichlet parameters (B, C, H, C, C)
#         updated_dirichlets = batch_update_dirichlet_for_item_hard(
#             dirichlet_alphas, chunk_preds, C
#         )
        
#         # Convert to Beta parameters (B, C, H)
#         alpha_beta_batch, beta_beta_batch = dirichlet_to_beta_with_marginal(
#             updated_dirichlets.view(-1, C, C),
#             marginal_distribution
#         ).view(len(chunk), C, H)
        
#         # Compute updated probabilities (B, C, H)
#         updated_probs = sample_is_best_worker_beta_batched(
#             alpha_beta_batch, beta_beta_batch
#         )
        
#         # Compute entropy differences (B, C)
#         H_updated = distribution_entropy(updated_probs)
#         eig_chunk = H_current - H_updated
        
#         # Marginalize using class priors (B,)
#         eig_chunk = (eig_chunk * marginal_distribution).sum(dim=1)
#         eig_results.append(eig_chunk)
    
#     return torch.cat(eig_results)


def batch_update_dirichlet_for_item_soft(
    dirichlet_alphas: torch.Tensor,  # (H, C, C)
    worker_probs: torch.Tensor,      # (N, H, C) - soft predictions
    C: int,
    update_weight: float = 1.0
) -> torch.Tensor:
    """
    Soft update of Dirichlet parameters using probabilistic worker predictions
    """

    N, H, _ = worker_probs.shape
    
    # 1. Expand base parameters with correct dimensions (N, C, H, C, C)
    updated = dirichlet_alphas.unsqueeze(0).unsqueeze(1).expand(N, C, H, C, C).clone()  # (N, C, H, C, C)

    # 2. Create broadcastable update tensor (N, C, H, C)
    #    - Each true class c gets full worker_probs for all predicted classes
    updates = worker_probs.unsqueeze(1).expand(-1, C, -1, -1) * update_weight

    # 3. Add updates to corresponding confusion matrix rows
    #    Use advanced indexing to target [:, c, :, c, :] slices
    for c in range(C):
        updated[:, c, :, c, :] += updates[:, c, :, :]

    return updated

import torch
import torch.nn.functional as F

def sample_is_best_worker_mixture_batched(
    alpha_batch: torch.Tensor,  # (B, C, H)
    beta_batch:  torch.Tensor,  # (B, C, H)
    pi_c:        torch.Tensor,  # (C,)
    num_points:  int = 128,
    eps:         float = 1e-30,
    chunk_size:  int = None
) -> torch.Tensor:
    """
    Computes P(worker h is 'best') for each item & hypothetical class scenario,
    but each worker's "accuracy distribution" is a mixture of Beta over classes.

    alpha_batch, beta_batch: shape (B, C, H)
      - B = number of items in the batch (e.g. your candidate x class combos)
      - C = number of classes
      - H = number of workers/models
    pi_c: shape (C,), the class prior for mixing.

    Returns:
      prob_best_out: shape (B, C, H)
        i.e., for each of the B items and each "class scenario" c in [0..C-1],
        we get a distribution over H workers (who is best).

      In typical usage, you might only interpret the (B, C) portion as "each item & each
      hypothetical true class," so the final dimension H is which worker is best.

    Implementation outline:
      1) We build mixture_pdf[b,h,x] = sum_c pi_c[c] * BetaPDF(alpha_batch[b,c,h], beta_batch[b,c,h], x).
      2) We build mixture_cdf likewise.
      3) Probability that worker h is best = ∫ mixture_pdf[b,h,x] * ∏_{k != h} mixture_cdf[b,k,x] dx.
      4) We do chunking in B if needed.
    """
    device = alpha_batch.device
    B, C, H = alpha_batch.shape

    if chunk_size is None:
        chunk_size = B

    # We'll do a trapezoid over x in [0..1].
    eps=1e-6
    x = torch.linspace(eps, 1.0-eps, steps=num_points, device=device)
    
    # Prepare output => shape (B, C, H)
    prob_best_out = torch.zeros(B, C, H, device=device)
    
    start = 0
    while start < B:
        end = min(start + chunk_size, B)
        batch_size = end - start

        alpha_ch = alpha_batch[start:end]  # (batch_size, C, H)
        beta_ch  = beta_batch[start:end]   # (batch_size, C, H)
        
        # We want a mixture PDF/CDF for each (batch_item,h).
        # We'll do it by building Beta PDFs for each (batch_item, c, h) => then sum over c w.r.t. pi_c.

        # 1) Build a grid-based Beta PDF for each (bInChunk, c, h).
        # We'll flatten => shape (batch_size*C*H,) for a single distribution call
        alpha_flat = alpha_ch.reshape(-1)
        beta_flat  = beta_ch.reshape(-1)
        
        # Evaluate PDFs at each x => shape (num_points, batch_size*C*H)
        dist = torch.distributions.Beta(alpha_flat, beta_flat)
        log_pdf_vals = dist.log_prob(x.unsqueeze(1))  # (num_points, batch_size*C*H)
        pdf_vals_all = log_pdf_vals.exp()             # (num_points, batch_size*C*H)
        
        # Reshape => (batch_size, C, H, num_points)
        pdf_vals_all = pdf_vals_all.transpose(0,1).reshape(batch_size, C, H, num_points)
        
        # 2) Build partial CDF by trapezoid integration of the PDF
        cdf_vals_all = torch.zeros_like(pdf_vals_all)  # (batch_size, C, H, num_points)
        for j in range(1, num_points):
            dx = x[j] - x[j-1]
            trap = 0.5*(pdf_vals_all[:,:,:,j] + pdf_vals_all[:,:,:,j-1]) * dx
            cdf_vals_all[:,:,:,j] = cdf_vals_all[:,:,:,j-1] + trap
        
        # 3) Sum across c => mixture
        # mixture_pdf[b,h,x] = sum_{c} pi_c[c] * pdf_vals_all[b,c,h,x]
        # mixture_cdf[b,h,x] = sum_{c} pi_c[c] * cdf_vals_all[b,c,h,x]
        pi_bc = pi_c.view(1, C, 1, 1)  # (1,C,1,1)
        mixture_pdf = (pdf_vals_all * pi_bc).sum(dim=1)  # => (batch_size, H, num_points)
        mixture_cdf = (cdf_vals_all * pi_bc).sum(dim=1)  # => (batch_size, H, num_points)
        
        # 4) Probability worker h is best => integrand[h,x] = mixture_pdf[h,x]*∏_{k≠h} mixture_cdf[k,x]
        mixture_cdf_clamped = mixture_cdf.clamp_min(eps)
        log_cdf = mixture_cdf_clamped.log()  # => (batch_size, H, num_points)
        sum_all = log_cdf.sum(dim=1, keepdim=True)  # => (batch_size,1,num_points)
        log_excl = sum_all - log_cdf            # => (batch_size,H,num_points)
        product_excl = torch.exp(log_excl)      # => (batch_size,H,num_points)
        
        integrand = mixture_pdf * product_excl  # => (batch_size,H,num_points)
        
        prob_best_chunk = torch.trapz(integrand, x, dim=2)  # => (batch_size,H)
        
        # 5) We want shape (batch_size, C, H).
        # We typically interpret the "C" dimension as each hypothetical true class.
        # So we replicate prob_best_chunk across the 'C' dimension:
        prob_best_chunk_rep = prob_best_chunk.unsqueeze(1).expand(-1, C, -1)  # => (batch_size, C, H)
        
        # (Optional) Normalize over H => so sum(prob_best_chunk_rep[h])=1 for each (b,c).
        denom = prob_best_chunk_rep.sum(dim=2, keepdim=True).clamp_min(eps)
        prob_best_chunk_rep = prob_best_chunk_rep / denom
        
        # 6) Store in final
        prob_best_out[start:end] = prob_best_chunk_rep
        
        start = end

    return prob_best_out


def eig_dirichlet_batched_soft(
    dirichlet_alphas: torch.Tensor,
    worker_probs: torch.Tensor,
    marginal_distribution: torch.Tensor,
    candidate_ids: list[int],
    chunk_size: int = 100,
    update_weight = 1.0,
    update_rule="soft"
) -> torch.Tensor:
    device = dirichlet_alphas.device
    candidates = torch.tensor(candidate_ids, device=device)
    C = dirichlet_alphas.size(1)  # Number of classes
    
    # Convert to Beta approximation
    # alpha_beta, beta_beta = dirichlet_to_beta_with_marginal(dirichlet_alphas, marginal_distribution)
    # current_probs = sample_is_best_worker(alpha_beta, beta_beta)
    current_probs = sample_is_best_worker_dirichlet(dirichlet_alphas, marginal_distribution)
    
    print("model 0 dirichlet 0 sum", dirichlet_alphas[0][0].sum())
    print("marginal sum", marginal_distribution.sum())
    print("current probs sum", current_probs.sum())

    H_current = distribution_entropy(current_probs)
    print("H current", H_current)
    
    eig_results = []
    for chunk_start in tqdm(range(0, len(candidates), chunk_size)):
        chunk = candidates[chunk_start:chunk_start+chunk_size]
        chunk_probs = worker_probs[chunk]
        
        if update_rule == 'hard':
            chunk_probs = F.one_hot(torch.argmax(chunk_probs, dim=-1), num_classes=C).float()

        # Updated dirichlets shape: (B, C, H, C, C)
        updated_dirichlets = batch_update_dirichlet_for_item_soft(
            dirichlet_alphas, 
            chunk_probs, 
            C, 
            update_weight=update_weight
        )
        if chunk_start == 0:
            print("model 0 dirichlet 0 sum after batch update", updated_dirichlets[0][0][0][0].sum())

        B, candidate_C, H, C1, C2 = updated_dirichlets.shape  # note: candidate_C should equal C
        # Flatten the candidate and model dimensions together:
        flat_dirichlets = updated_dirichlets.reshape(B * candidate_C * H, C1, C2)  # shape: (B * C * H, C, C)

        # Convert each confusion matrix (of shape (C, C)) to a Beta distribution using your conversion function.
        alpha_beta_flat, beta_beta_flat = dirichlet_to_beta_with_marginal(flat_dirichlets, marginal_distribution)
        # These have shape (B * C * H,)

        # Now reshape the output back to (B, candidate_C, H)
        alpha_beta_batch = alpha_beta_flat.reshape(B, candidate_C, H)
        beta_beta_batch = beta_beta_flat.reshape(B, candidate_C, H)
        
        if chunk_start == 0:
            print("model 0 class 0 alpha beta after reshape", alpha_beta_batch[0][0][0], beta_beta_batch[0][0][0])
            print("alpha_beta_batch.shape", alpha_beta_batch.shape, beta_beta_batch.shape)
            print("nan check", torch.any(torch.isnan(alpha_beta_batch)), torch.any(torch.isnan(beta_beta_batch)))
            print("negative check", torch.any(alpha_beta_batch < 0), torch.any(beta_beta_batch < 0))

        # Rest of the workflow remains the same
        updated_probs = sample_is_best_worker_beta_batched(
            alpha_beta_batch, 
            beta_beta_batch
        ) # B, C, H
        if chunk_start == 0:
            print("updated_probs.shape", updated_probs.shape)
            print("updated probs N=0, hypothetical C=0", updated_probs[0][0])
            print("updated probs sum", updated_probs[0][0].sum())
            print("updated probs nan check", torch.any(torch.isnan(updated_probs)))
        
        # H_updated = distribution_entropy(updated_probs)
        p_clamped = updated_probs.clamp_min(1e-12) # B, C, H
        H_updated = -(p_clamped * p_clamped.log2()).sum(dim=-1) # B, C ?
        
        if chunk_start == 0:
            print("H udpated shape", H_updated.shape)
            print("H updated", H_updated[0])

        eig_chunk = (H_current - H_updated) * marginal_distribution.unsqueeze(0)
        eig_results.append(eig_chunk.sum(dim=1))
    
    return torch.cat(eig_results)


def sample_is_best_worker_mixture_batched_2D(
    alpha_reshaped: torch.Tensor,  # (B*C, H, C)
    beta_reshaped:  torch.Tensor,  # (B*C, H, C)
    pi_c: torch.Tensor,            # (C,)
    num_points: int = 128,
    eps: float = 1e-30
) -> torch.Tensor:
    """
    For each batch item i in [0..B*C-1], we have:
      alpha_reshaped[i, h, c], beta_reshaped[i, h, c],
    meaning model h has Beta(alpha, beta) for class c, with mixture weight pi_c[c].
    We build a mixture distribution for each (i, h):
      f_{i,h}(x) = sum_{c} pi_c[c]*BetaPDF(alpha_{i,h,c}, beta_{i,h,c}, x)
      F_{i,h}(x) = sum_{c} pi_c[c]*BetaCDF(...)
    Then the probability that model h is best is:
      \int f_{i,h}(x) * prod_{k != h} F_{i,k}(x) dx
    Return shape: (B*C, H).

    This is a "2D" version because we treat the first dimension (B*C) as
    the "batch items," second dimension H as the number of workers, and
    third dimension C as the mixture components.
    """

    device = alpha_reshaped.device
    BC, H, C = alpha_reshaped.shape  # "BC" stands for B*C

    # 1) Create an x-grid
    eps = 1e-6
    x_grid = torch.linspace(eps, 1.0-eps, steps=num_points, device=device)
    dx = x_grid[1:] - x_grid[:-1]   # for partial trapezoids

    # We'll store PDFs and CDFs in shape (BC, H, num_points)
    mixture_pdf = torch.zeros(BC, H, num_points, device=device)
    mixture_cdf = torch.zeros(BC, H, num_points, device=device)

    # 2) We'll do a loop over c or vector expansions. A simple approach:
    for c_idx in range(C):
        alpha_c = alpha_reshaped[..., c_idx]  # shape (BC, H)
        beta_c  = beta_reshaped[..., c_idx]   # shape (BC, H)
        weight_c = pi_c[c_idx]

        # Build Beta PDF for each (BC,H). We'll flatten => (BC*H) then reshape.
        alpha_flat = alpha_c.view(-1)
        beta_flat  = beta_c.view(-1)

        dist = torch.distributions.Beta(alpha_flat, beta_flat)
        # Evaluate PDF at shape (num_points, BC*H)
        logpdf = dist.log_prob(x_grid.unsqueeze(1))  # => (num_points, BC*H)
        pdf_c   = torch.exp(logpdf).transpose(0,1).view(BC, H, num_points)
        # cdf_c: we do a manual trapezoid
        cdf_c = torch.zeros_like(pdf_c)
        for j in range(1, num_points):
            dxx = x_grid[j] - x_grid[j-1]
            trap = 0.5*(pdf_c[:,:,j] + pdf_c[:,:,j-1]) * dxx
            cdf_c[:,:,j] = cdf_c[:,:,j-1] + trap

        # Weighted by pi_c[c_idx]
        mixture_pdf += weight_c * pdf_c
        mixture_cdf += weight_c * cdf_c

    # 3) Probability of "worker h is best": integrand[h,x] = mixture_pdf[h,x] * ∏_{k!=h} mixture_cdf[k,x]
    # We'll do it in log-space for numeric stability
    mixture_cdf = mixture_cdf.clamp_min(eps)
    log_cdf = mixture_cdf.log()  # (BC,H,num_points)
    sum_log_cdfs = log_cdf.sum(dim=1, keepdim=True)  # (BC,1,num_points)
    # product_excl[h,x] = exp( sum_log_cdfs[x] - log_cdf[h,x] )
    # => shape (BC,H,num_points)
    log_excl = sum_log_cdfs - log_cdf
    product_excl = torch.exp(log_excl)

    integrand = mixture_pdf * product_excl  # => (BC,H,num_points)

    # 4) Trapezoidal integration over x => (BC,H)
    prob_best = torch.zeros(BC, H, device=device)
    for j in range(1, num_points):
        dxx = x_grid[j] - x_grid[j-1]
        trap = 0.5*(integrand[:,:,j] + integrand[:,:,j-1]) * dxx
        prob_best += trap

    # (Optional) normalize over H for each item i => sum(prob_best[i,h])=1
    denom = prob_best.sum(dim=1, keepdim=True).clamp_min(eps)
    prob_best = prob_best / denom  # => shape (BC,H)

    return prob_best

# more principled integral method - slower
# def eig_dirichlet_batched_soft_new(
#     dirichlet_alphas: torch.Tensor,
#     worker_probs: torch.Tensor,
#     marginal_distribution: torch.Tensor,
#     candidate_ids: list[int],
#     chunk_size: int = 100,
#     update_weight = 1.0,
#     update_rule="soft"
# ) -> torch.Tensor:
#     device = dirichlet_alphas.device
#     candidates = torch.tensor(candidate_ids, device=device)
#     C = dirichlet_alphas.size(1)  # Number of classes
    
#     # "Current" best-model distribution: we do the mixture approach
#     current_probs = sample_is_best_worker_dirichlet(dirichlet_alphas, marginal_distribution)
#     # (Or you might do sample_is_best_worker_class_conditional_mixture(...) if you have such a single-item method)
    
#     print("model 0 dirichlet 0 sum", dirichlet_alphas[0][0].sum())
#     print("marginal sum", marginal_distribution.sum())
#     print("current probs sum", current_probs.sum())

#     H_current = distribution_entropy(current_probs)
#     print("H current", H_current)
    
#     eig_results = []
#     for chunk_start in tqdm(range(0, len(candidates), chunk_size)):
#         chunk = candidates[chunk_start:chunk_start+chunk_size]
#         chunk_probs = worker_probs[chunk]
        
#         if update_rule == 'hard':
#             chunk_probs = F.one_hot(torch.argmax(chunk_probs, dim=-1), num_classes=C).float()

#         # Updated dirichlets shape: (B, C, H, C, C)
#         updated_dirichlets = batch_update_dirichlet_for_item_soft(
#             dirichlet_alphas, 
#             chunk_probs, 
#             C, 
#             update_weight=update_weight
#         )
#         B, candidate_C, H, _, _ = updated_dirichlets.shape  # candidate_C==C

#         # Flatten => shape (B*C*H, C, C)
#         flat_dirichs = updated_dirichlets.reshape(B * candidate_C * H, C, C)
        
#         # Convert each (C,C) confusion matrix to a "per-class Beta distribution" shape => (C,) for alpha/beta
#         # We'll do the same approach as "dirichlet_to_beta(...)", but for the diagonal vs off diagonal
#         # Then store them in shape (B*C, C, H).
#         alpha_list = []
#         beta_list  = []
#         for i in range(flat_dirichs.shape[0]):
#             # shape (C,C)
#             cmat = flat_dirichs[i]
#             diag = torch.diag(cmat)           # shape (C,)
#             row_sum = cmat.sum(dim=1)         # shape (C,)
#             alpha_c = diag
#             beta_c  = row_sum - diag
#             alpha_list.append(alpha_c.unsqueeze(0))
#             beta_list.append(beta_c.unsqueeze(0))
#         # cat => shape (B*C*H, 1, C)
#         alpha_stacked = torch.cat(alpha_list, dim=0)
#         beta_stacked  = torch.cat(beta_list, dim=0)
        
#         # We want shape => (B*C, C, H)
#         # But currently we have B*C*H rows. Actually we must re-group them by H, or adapt the approach.
#         # In your code, we flatten everything by (B * C * H). Let's reorganize:
#         # We had (B*C*H) "model" dimension in the flatten. Actually we want each model => separate confusion matrix.
#         # So let's reorder carefully:

#         # Actually, from your code we see:
#         # The dimension "B*C*H" is in 1 dimension: for item in B, for c in C, for h in H.
#         # So the index i => i // (H) => which item*c pair, i % H => which model.
#         # We'll pivot that shape into (B*C, H, C). Then we can pass it to the mixture function.

#         alpha_bc_h_c = alpha_stacked.view(B, candidate_C, H, C)  # shape (B,C,H,C)
#         beta_bc_h_c  = beta_stacked.view(B, candidate_C, H, C)   # shape (B,C,H,C)

#         # We want (B*C, C, H) or (B,C,H) => let's do (B*C,H,C) to feed the new function
#         # We'll reorder to (B*C, C, H) => i.e. gather the alpha for each item+class => a set of H distributions, each with C.
#         # Let's do: alpha_bc_h_c => (B, C, H, C), we permute => (B, C, H, C) => (B*C, H, C)?
#         alpha_reshaped = alpha_bc_h_c.permute(0,1,2,3).reshape(B*candidate_C, H, C)
#         beta_reshaped  = beta_bc_h_c.permute(0,1,2,3).reshape(B*candidate_C, H, C)

#         # Now we can call our mixture approach => returns shape (B*C, H),
#         # but we want (B, C, H). We'll do the final reshape after the call.

#         # We'll define a new function "sample_is_best_worker_mixture_batched_2D"
#         # that returns shape (B*C, H). Then we reshape => (B, C, H).

#         updated_probs_mixture = sample_is_best_worker_mixture_batched_2D(
#             alpha_reshaped, beta_reshaped,
#             marginal_distribution,  # pi_c
#             num_points=128,
#             eps=1e-30
#         ) # => shape (B*C, H)

#         updated_probs_mixture = updated_probs_mixture.view(B, candidate_C, H)  # => (B, C, H)

#         # Then the rest is the same: compute the new entropy, subtract from H_current, etc.
#         p_clamped = updated_probs_mixture.clamp_min(1e-30)
#         H_updated = -(p_clamped * p_clamped.log2()).sum(dim=-1)  # => (B, C)
#         eig_chunk = (H_current - H_updated) * marginal_distribution.unsqueeze(0)
#         eig_results.append(eig_chunk.sum(dim=1))

#     return torch.cat(eig_results)

def l1_dirichlet_batched(
    dirichlet_alphas: torch.Tensor,
    worker_probs: torch.Tensor,
    marginal_distribution: torch.Tensor,
    candidate_ids: list[int],
    chunk_size: int = 100,
    update_weight: float = 1.0,
    update_rule: str = "soft",
    weighted: bool = False
) -> torch.Tensor:
    """
    Computes a "weighted L1" difference between the current 'best-model' distribution
    and the updated 'best-model' distribution for each candidate item in candidate_ids.

    Analogous to your worker_diff_batched_scatter or eig_dirichlet_batched_soft, but
    instead of an EIG, we measure an L1 difference.

    Args:
        dirichlet_alphas: (H, C, C) - Dirichlet parameters for H models, each row c of confusion matrix.
        worker_probs: (N, H, C) - "soft" predictions for N items from H models, or one-hot if needed.
        marginal_distribution: (C,) - class prior probabilities
        candidate_ids: list of item indices we want to evaluate
        chunk_size: how many items to handle in each loop
        update_weight: scale factor when updating Dirichlet with the new item
        update_rule: "soft" or "hard" => whether we treat worker_probs as distributions or argmax
        weighted: if True, multiply the absolute difference by the current best-model probabilities

    Returns:
        A tensor of shape (len(candidate_ids),) with the L1 difference per candidate item.
    """
    device = dirichlet_alphas.device
    candidates = torch.tensor(candidate_ids, device=device)
    C = dirichlet_alphas.size(1)  # number of classes

    # 1) Compute the current "best model" distribution, shape (H,)
    #    i.e. probability each model is the best, given dirichlet_alphas.
    current_probs = sample_is_best_worker_dirichlet(
        dirichlet_alphas,      # (H, C, C)
        marginal_distribution  # (C,)
    )  # => shape (H,)

    # 2) We'll store results in a vector for each candidate
    M = len(candidate_ids)
    results = torch.zeros(M, device=device)

    # 3) Process candidates in chunks
    for chunk_start in tqdm(range(0, M, chunk_size)):
        chunk = candidates[chunk_start : chunk_start + chunk_size]
        chunk_probs = worker_probs[chunk]  # shape (B, H, C), B = chunk_size

        # Possibly convert to one-hot if 'hard' rule
        if update_rule == "hard":
            # argmax across classes => one-hot
            chunk_probs = F.one_hot(torch.argmax(chunk_probs, dim=-1), num_classes=C).float()
            # shape (B, H, C)

        # 4) Hypothetically update the Dirichlet with each "true class" c in [0..C-1]
        #    batch_update_dirichlet_for_item_soft => shape (B, C, H, C, C)
        updated_dirichlets = batch_update_dirichlet_for_item_soft(
            dirichlet_alphas,
            chunk_probs,  # (B, H, C)
            C,
            update_weight=update_weight
        )
        # => updated_dirichlets: (B, C, H, C, C)

        # 5) Flatten so we can easily do Dirichlet->Beta conversions
        # B, candidate_C, H, _, _ = updated_dirichlets.shape
        # flat_dirichlets = updated_dirichlets.view(B * candidate_C * H, C, C)
        B, candidate_C, H, C1, C2 = updated_dirichlets.shape  # note: candidate_C should equal C
        # Flatten the candidate and model dimensions together:
        flat_dirichlets = updated_dirichlets.reshape(B * candidate_C * H, C1, C2)  # shape: (B * C * H, C, C)

        # 6) Convert each confusion matrix to Beta for overall accuracy
        alpha_beta_flat, beta_beta_flat = dirichlet_to_beta_with_marginal(
            flat_dirichlets,        # shape (B*C*H, C, C)
            marginal_distribution   # (C,)
        )  # => shapes (B*C*H,), (B*C*H,)

        # Reshape alpha,beta back to (B, C, H)
        alpha_beta_batch = alpha_beta_flat.view(B, candidate_C, H)
        beta_beta_batch  = beta_beta_flat.view(B, candidate_C, H)

        # 7) Compute the "best-worker" distribution for each hypothetical update
        updated_probs = sample_is_best_worker_beta_batched(
            alpha_beta_batch,  # (B, C, H)
            beta_beta_batch    # (B, C, H)
        )  # => shape (B, C, H)

        # 8) Now compute the L1 difference
        #    current_probs: shape (H,)
        #    updated_probs[b, c, h]: shape (B, C, H)
        # We'll broadcast current_probs to (B, C, H) with unsqueeze(0,1)
        current_probs_expanded = current_probs.unsqueeze(0).unsqueeze(1)  # (1,1,H)

        differences = torch.abs(updated_probs - current_probs_expanded)  # (B, C, H)

        if weighted:
            # multiply by the current best-model distribution for weighting
            differences = differences * current_probs_expanded

        # Sum over the model dimension => shape (B, C)
        diffs = differences.sum(dim=-1)

        # 9) Weight by the class prior distribution, sum over classes => shape (B,)
        diffs_weighted = (marginal_distribution.unsqueeze(0) * diffs).sum(dim=1)

        results[chunk_start : chunk_start + chunk_size] = diffs_weighted

    return results


class CODA(ModelSelector):
    def __init__(self, dataset,
                prior_source="ens",
                q='eig',                       # acquisition function
                prefilter_fn='disagreement',   # filter with heuristic
                prefilter_n=0,                 # number to filter down to
                epsilon=0.0,                   
                update_rule="hard",
                base_prior="diag",

                # deprecating
                hypothetical_update_strength=None, #1.0,
                prior_strength=None, #0.1,
                base_strength=None, #1.0,
                update_strength=None, #0.1,

                # new
                temperature=1.0,
                alpha=0.9,
                learning_rate_ratio=0.01
                 ):
        self.dataset = dataset
        self.device = dataset.pred_logits.device
        self.H, self.N, self.C = dataset.pred_logits.shape
        self.q = q
        self.prefilter_fn = prefilter_fn
        self.prefilter_n = prefilter_n
        self.epsilon = epsilon
        self.update_rule = update_rule
        self.base_prior = base_prior

        # OLD
        # self.base_strength = base_strength
        # self.prior_strength = prior_strength
        # self.update_strength = update_strength
        # self.hypothetical_update_strength = hypothetical_update_strength

        # NEW
        self.base_strength = alpha / temperature
        self.prior_strength = (1 - alpha) / temperature
        self.update_strength = learning_rate_ratio / temperature
        self.hypothetical_update_strength = 1.0 # fix this for sanity

        # initialize dirichlets (confusion matrices) and class marginals
        ensemble_preds = torch.argmax(Ensemble(dataset.pred_logits).get_preds(), dim=-1)
        pred_classes = torch.argmax(dataset.pred_logits, dim=-1)  # (H, N)
        if prior_source == "ens" or prior_source == "ens-exp":
            preds_soft = F.softmax(dataset.pred_logits, dim=-1)
            soft_confusion = create_confusion_matrices(true_labels=ensemble_preds, 
                                                       model_predictions=preds_soft, 
                                                       mode='soft')
            self.dirichlets = initialize_dirichlets_with_prior(soft_confusion, 
                                                               self.prior_strength,
                                                               base_strength=self.base_strength,
                                                               base_prior=self.base_prior)
            self.curr_marginal = compute_ensemble_marginal(self.dirichlets, preds_soft)
            self.curr_marginal = self.curr_marginal / self.curr_marginal.sum()
        # elif prior_source == "ens-scaled":
        #     preds_soft = F.softmax(dataset.pred_logits, dim=-1)
        #     soft_confusion = create_confusion_matrices(true_labels=ensemble_preds, 
        #                                                model_predictions=preds_soft, 
        #                                                mode='soft',
        #                                                base_prior=self.base_prior)
        #     self.dirichlets = initialize_dirichlets_with_prior(soft_confusion, self.prior_strength / (self.C-1))
        #     self.curr_marginal = compute_ensemble_marginal(self.dirichlets, preds_soft)
        #     self.curr_marginal = self.curr_marginal / self.curr_marginal.sum()
        # elif prior_source == "ens-hard":
        #     pred_one_hot = F.one_hot(pred_classes, num_classes=self.C).float()  # (H, N, C)
        #     hard_confusion = create_confusion_matrices(true_labels=ensemble_preds, 
        #                                                model_predictions=pred_one_hot, 
        #                                                mode='hard',
        #                                                base_prior=self.base_prior)
        #     self.dirichlets = initialize_dirichlets_with_prior(hard_confusion, self.prior_strength)
        #     self.curr_marginal = compute_ensemble_marginal(self.dirichlets, pred_one_hot)
        #     self.curr_marginal = self.curr_marginal / self.curr_marginal.sum()
        else:
            raise NotImplemented

        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(dataset.pred_logits.shape[1]))
        self.qms = []

    @classmethod
    def from_args(cls, dataset, args):
        """Init from command line args. See main.py."""
        return cls(dataset,
                    prior_source=args.prior_source, 
                    q=args.q,
                    prefilter_fn=args.prefilter_fn,
                    prefilter_n=args.prefilter_n,
                    epsilon=args.epsilon,
                    update_rule=args.update_rule,
                    # update_strength=args.update_strength,
                    # prior_strength=args.prior_strength,
                    # base_strength=args.base_strength,
                    # hypothetical_update_strength=args.hypothetical_update_strength,
                    temperature=args.temperature,
                    alpha=args.alpha,
                    learning_rate_ratio=args.learning_rate_ratio,
                    base_prior=args.base_prior
                )

    def get_next_item_to_label(self):
        pred_classes = torch.argmax(self.dataset.pred_logits, dim=-1)

        _d_u_idxs = self.d_u_idxs
        if self.prefilter_fn == 'disagreement':
            majority_class, _ = torch.mode(pred_classes, dim=0)
            not_majority_mask = (pred_classes != majority_class.unsqueeze(0))
            num_disagrements = not_majority_mask.sum(dim=0) # => (N,)
            idxs_with_disagreement = num_disagrements.nonzero(as_tuple=True)[0].tolist()
            remaining = set(self.d_u_idxs)
            _d_u_idxs = [idx for idx in idxs_with_disagreement if idx in remaining]
        elif self.N > self.prefilter_n and self.prefilter_fn is not None:
            if self.prefilter_fn == 'iid':
                _d_u_idxs = random.sample(self.d_u_idxs, k=self.prefilter_n)
            else:
                raise NotImplemented

        if len(_d_u_idxs) > self.prefilter_n and self.prefilter_n > 0:
            _d_u_idxs = random.sample(_d_u_idxs, k=self.prefilter_n)

        q = self.q

        if len(_d_u_idxs) == 0:
            print("Prefiltered all points; falling back to random sampling")
            q = 'iid'

        if self.epsilon and random.random() < self.epsilon:
            print("Epsilon sample")
            q = 'iid'

        if q == 'iid':
            _q_values = torch.ones(len(_d_u_idxs), device=self.device) * 1 / len(_d_u_idxs)
        elif q =='eig':
            dataset_tensor = F.softmax(self.dataset.pred_logits.permute(1,0,2), dim=-1)
            _q_values = eig_dirichlet_batched_soft(
                                self.dirichlets, 
                                dataset_tensor,
                                self.curr_marginal,
                                _d_u_idxs,
                                chunk_size=4,
                                update_rule=self.update_rule,
                                update_weight=self.hypothetical_update_strength
            )
        elif q == 'weighted_l1':
            dataset_tensor = F.softmax(self.dataset.pred_logits.permute(1,0,2), dim=-1)
            _q_values = l1_dirichlet_batched(
                                self.dirichlets, 
                                dataset_tensor,
                                self.curr_marginal,
                                _d_u_idxs,
                                chunk_size=4,
                                update_rule=self.update_rule,
                                update_weight=self.hypothetical_update_strength,
                                weighted=True
            )
        elif q == 'uncertainty':
            pred_probs = F.softmax(self.dataset.pred_logits, dim=2)
            mean_pred_probs = pred_probs.mean(dim=0)
            epsilon = 1e-8
            entropy_per_data_point = -torch.sum(mean_pred_probs * torch.log(mean_pred_probs + epsilon), dim=-1)
            entropy_per_unlabeled_data_point = entropy_per_data_point[_d_u_idxs]
            _q_values = entropy_per_unlabeled_data_point
            # chosen_q, chosen_idx_local = torch.max(entropy_per_unlabeled_data_point, dim=0)
            # chosen_idx_global = self.d_u_idxs[chosen_idx_local]
            # return chosen_idx_global, chosen_q

        chosen_q_val, chosen_local_idx = torch.max(_q_values, dim=0)
        print("chosen_q_val, chosen_local_idx", chosen_q_val, chosen_local_idx)
        ties = (_q_values == chosen_q_val)
        print("ties", ties)
        if ties.sum() > 1:
            print(ties.sum(), "ties at Q=", chosen_q_val, "; randomly selecting one of these")
            idxs = torch.nonzero(ties, as_tuple=True)[0]
            chosen_local_idx = idxs[torch.randperm(len(idxs))[0]]
        # chosen_idx = self.d_u_idxs[chosen_local_idx.item()]
        chosen_idx = _d_u_idxs[chosen_local_idx.item()]
        chosen_q = chosen_q_val

        print("Chosen q, candidate index:", chosen_q, chosen_idx)
        return chosen_idx, chosen_q
    
    def add_label(self, chosen_idx, true_class, selection_prob):
        true_class = int(true_class) # just in case; e.g. GLUE preds break this
        preds_soft = F.softmax(self.dataset.pred_logits, dim=-1)
        preds_on_item = preds_soft[:, chosen_idx, :]
        if self.update_rule == "hard":
            preds_on_item = F.one_hot(torch.argmax(preds_on_item, dim=-1), num_classes=self.C).float()

        self.dirichlets[:, true_class, :] = self.dirichlets[:, true_class, :] + self.update_strength * preds_on_item
        
        # update class marginal
        self.curr_marginal = compute_ensemble_marginal(self.dirichlets, preds_soft)
        self.curr_marginal = self.curr_marginal / self.curr_marginal.sum()

        self.d_l_idxs.append(chosen_idx)
        self.d_u_idxs.remove(chosen_idx)
        self.qms.append(selection_prob)
        self.d_l_ys.append(true_class)

    def get_best_model_prediction(self):
        prob_best = sample_is_best_worker_dirichlet(self.dirichlets, self.curr_marginal)

        if torch.isnan(prob_best).any():
            raise ValueError("NaN in posterior")

        best_model_idx_pred = torch.argmax(prob_best)
        print(f"After update, best model index:", best_model_idx_pred, prob_best.max())
        print("Top 10:", torch.topk(prob_best, k=min(10, self.H))[0]) 

        return best_model_idx_pred