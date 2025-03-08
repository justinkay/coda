import torch
from torch.distributions import Beta
from tqdm import tqdm
import random

import metrics
from surrogates import Ensemble, WeightedEnsemble, DawidSkeneModel
from ams.base import ModelSelector


def sample_is_best_worker(alpha: torch.Tensor,
                          beta: torch.Tensor,
                          num_draws=1000,
                          random_state=0,
                          num_points=128) -> torch.Tensor:
    """
    Given alpha[k], beta[k] for K workers (Beta posterior parameters) as PyTorch tensors,
    *now* compute the probability that each worker is best using a 1D integral
    (instead of Monte Carlo).

    alpha, beta: shape (K,) on the same device (CPU or CUDA).
    Returns prob_best: shape (K,) on the same device.

    (num_draws, random_state are no longer used.)
    """
    device = alpha.device
    K = alpha.shape[0]

    # We'll use a simple grid approach
    x = torch.linspace(0.0, 1.0, num_points, device=device)

    # Build Beta distributions
    dist_betas = torch.distributions.Beta(alpha, beta)  # batch-size = K
    pdf_vals = torch.exp(dist_betas.log_prob(x.unsqueeze(1)))  # shape (num_points, K)
    pdf_vals = pdf_vals.transpose(0, 1)  # => (K, num_points)

    # Approximate CDF of each Beta by trapezoidal integration of PDF
    cdf_vals = torch.zeros_like(pdf_vals)  # shape (K, num_points)
    for j in range(1, num_points):
        dx = x[j] - x[j - 1]
        trap = 0.5 * (pdf_vals[:, j] + pdf_vals[:, j - 1]) * dx
        cdf_vals[:, j] = cdf_vals[:, j - 1] + trap

    # (Optional) normalize CDF so it ends at 1 at x=1
    # cdf_vals = cdf_vals / cdf_vals[:, -1:].clamp_min(1e-30)

    # We'll clamp to avoid log(0) if needed
    cdf_vals_clamped = cdf_vals.clamp_min(1e-30)
    log_cdfs = torch.log(cdf_vals_clamped)  # shape (K, num_points)

    # sum of log_cdfs across all workers => shape(num_points,)
    sum_log_cdfs = log_cdfs.sum(dim=0)  # (num_points,)

    # Probability that worker i is best:
    # Integrand_i(x) = pdf_i(x) * ∏_{j != i} cdf_j(x)
    # => log( ∏_{j != i} cdf_j(x)) = sum_{j} log cdf_j(x) - log cdf_i(x)
    log_exclusive = sum_log_cdfs.unsqueeze(0) - log_cdfs  # shape (K, num_points)
    product_exclusive = torch.exp(log_exclusive)          # shape (K, num_points)

    integrand = pdf_vals * product_exclusive  # shape (K, num_points)

    # Final integration -> shape(K,)
    prob_best = torch.trapz(integrand, x, dim=1)

    # normalize
    prob_best = prob_best / prob_best.sum()

    return prob_best


def batch_update_beta_for_item(
    alpha: torch.Tensor,  # shape (K,)
    beta: torch.Tensor,   # shape (K,)
    worker_preds: torch.Tensor,  # shape (K,) => worker_preds for item i
    C: int,
    update_weight = 1.0
):
    """
    Create 'alpha_batch' and 'beta_batch' of shape (C, K).
    For each class c, worker k is correct if worker_preds[k] == c => alpha_batch[c,k] = alpha[k]+1
    else => beta_batch[c,k] = beta[k]+1.
    Everything else remains the same.

    Returns:
      alpha_batch, beta_batch: shape (C, K)
    """
    K_ = alpha.shape[0]

    # Start with copies of alpha, beta for each class => shape (C, K)
    alpha_batch = alpha.unsqueeze(0).expand(C, K_).clone()
    beta_batch  = beta.unsqueeze(0).expand(C, K_).clone()

    pred_classes = worker_preds.unsqueeze(0).expand(C, K_)  # replicate rowwise
    class_range  = torch.arange(C, device=alpha.device).unsqueeze(1).expand(C, K_)
    eq_mask = (pred_classes == class_range)  # shape (C, K)

    alpha_batch[eq_mask] += 1.0 * update_weight
    beta_batch[~eq_mask]  += 1.0 * update_weight

    return alpha_batch, beta_batch


def sample_is_best_worker_beta_batched(
    alpha_batch: torch.Tensor,   # shape (C,K) or (N,C,K)
    beta_batch:  torch.Tensor,   # same shape
    num_points:  int = 128,
    eps: float   = 1e-30,
    chunk_size:  int = None
) -> torch.Tensor:
    """
    Computes P(worker k is 'best') for each item & each class,
    via numeric integration of Beta PDFs on a grid of size `num_points`.

    - If alpha_batch,beta_batch have shape (C,K), returns shape (C,K)
      (i.e. a single item with C possible classes).
    - If alpha_batch,beta_batch have shape (N,C,K), returns shape (N,C,K).

    Arguments:
      alpha_batch: (C,K) or (N,C,K)
      beta_batch:  same shape
      num_points:  number of trapezoid steps in [0,1]
      eps:         clamp to avoid log(0)
      chunk_size:  process items in chunks if N is large

    Returns:
      prob_best_out:
        - shape (C,K) if input was (C,K)
        - shape (N,C,K) if input was (N,C,K)
    """
    device = alpha_batch.device

    # 1) If we only have (C,K), treat it as a single item => (1,C,K)
    if alpha_batch.ndim == 2:
        alpha_batch = alpha_batch.unsqueeze(0)  # => (1,C,K)
        beta_batch  = beta_batch.unsqueeze(0)   # => (1,C,K)
        single_item = True
    else:
        single_item = False

    N, C, K = alpha_batch.shape

    if chunk_size is None:
        chunk_size = N  # no chunking if not specified

    # We'll integrate over x in [0,1].
    # Reshape x to (num_points, 1) for proper broadcasting with batch dimensions
    grid_eps = 1e-6
    x = torch.linspace(grid_eps, 1.0-grid_eps, steps=num_points, device=device).unsqueeze(-1)  # shape (num_points, 1)

    # Prepare the output: (N,C,K)
    prob_best_out = torch.zeros(N, C, K, device=device)

    start = 0
    while start < N:
        end = min(start + chunk_size, N)
        batch_size = end - start

        # shape (batch_size,C,K)
        alpha_ch = alpha_batch[start:end]  
        beta_ch  = beta_batch[start:end]

        # Flatten => shape (batch_size*C, K)
        alpha_flat = alpha_ch.reshape(-1, K)
        beta_flat  = beta_ch.reshape(-1, K)

        # We have (batch_size*C) "items" x K workers => total Beta distributions = (batch_size*C*K).
        # But we want a Beta distribution per row & worker. Easiest approach:
        #    alpha_full => shape (batch_size*C*K,)
        alpha_full = alpha_flat.reshape(-1)
        beta_full  = beta_flat.reshape(-1)

        # Make a Beta distribution with batch_shape=(batch_size*C*K,).
        dist_full = Beta(alpha_full, beta_full)

        # Evaluate log_prob at each x => shape (num_points, batch_size*C*K)
        logpdf_full = dist_full.log_prob(x)  # shape (num_points, batch_size*C*K)

        # print("logpdf_full", torch.any(torch.isnan(logpdf_full).logical_or( torch.isinf(logpdf_full) )))

        # We want shape (batch_size*C*K, num_points), so we transpose:
        logpdf_full = logpdf_full.transpose(0, 1)  # => (batch_size*C*K, num_points)
        pdf_full = logpdf_full.exp()               # => (batch_size*C*K, num_points)

        # print("pdf_full", torch.any(torch.isnan(pdf_full).logical_or( torch.isinf(pdf_full) )))


        # Reshape => (batch_size*C, K, num_points)
        pdf_vals = pdf_full.reshape(batch_size*C, K, num_points)

        # 2) Trapezoid integration to get each worker's CDF
        cdf_vals = torch.zeros_like(pdf_vals)  # (batch_size*C, K, num_points)
        for j in range(1, num_points):
            dx = x[j] - x[j-1]
            trap = 0.5*(pdf_vals[:,:,j] + pdf_vals[:,:,j-1]) * dx
            cdf_vals[:,:,j] = cdf_vals[:,:,j-1] + trap

        # print("cdf_vals", torch.any(torch.isnan(cdf_vals).logical_or( torch.isinf(cdf_vals) )))

        cdf_clamped = cdf_vals.clamp_min(eps)
        log_cdfs = torch.log(cdf_clamped)  # => (batch_size*C, K, num_points)

        # print("log_cdfs", torch.any(torch.isnan(log_cdfs).logical_or( torch.isinf(log_cdfs) )))

        # sum over K => shape (batch_size*C, num_points)
        sum_log_cdfs = log_cdfs.sum(dim=1)

        # print("sum_log_cdfs", torch.any(torch.isnan(sum_log_cdfs).logical_or( torch.isinf(sum_log_cdfs) )))

        # Probability that worker k is best:
        # integrand_k(x) = pdf_k(x) * prod_{j!=k} cdf_j(x)
        # => log_exclusive_k(x) = sum_log_cdfs(x) - log_cdfs[k](x)
        log_exclusive = sum_log_cdfs.unsqueeze(1) - log_cdfs  # => (batch_size*C, K, num_points)

        # print('log exclusive max', log_exclusive.max())
        max_exponent = 30.0
        log_exclusive = log_exclusive.clamp_max(max_exponent)

        product_exclusive = torch.exp(log_exclusive)

        # print("log_exclusive", torch.any(torch.isnan(log_exclusive).logical_or( torch.isinf(log_exclusive) )))
        # print("product_exclusive", torch.any(torch.isnan(product_exclusive).logical_or( torch.isinf(product_exclusive) )))

        integrand = pdf_vals * product_exclusive  # => (batch_size*C, K, num_points)

        # print("integrand", torch.any(torch.isnan(integrand).logical_or( torch.isinf(integrand) )))

        # integrate wrt x => shape (batch_size*C, K)
        prob_best_flat = torch.trapz(integrand, x.squeeze(), dim=2)  # x.squeeze() to match original shape (num_points,)

        # print("prob_best_flat", torch.any(torch.isnan(prob_best_flat)))

        # reshape => (batch_size, C, K)
        prob_best_chunk = prob_best_flat.reshape(batch_size, C, K)

        # print("prob_best_chunk1 ", torch.any(torch.isnan(prob_best_chunk)))
        # print(prob_best_chunk)

        # normalize
        prob_best_chunk = prob_best_chunk / prob_best_chunk.sum(dim=-1, keepdim=True).clamp_min(eps)

        
        # print("prob_best_chunk bottom", torch.any(torch.isnan(prob_best_chunk.sum(dim=-1, keepdim=True).clamp_min(eps))))
        # print("prob_best_chunk2 ", torch.any(torch.isnan(prob_best_chunk)))

        # Store
        prob_best_out[start:end] = prob_best_chunk

        start = end

    # If single_item => shape (C,K)
    if single_item:
        return prob_best_out[0]

    return prob_best_out


def distribution_entropy(prob: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """
    Compute entropy of a discrete probability distribution 'prob' shape (K,).
    H(p) = - sum_k p_k * log(p_k).
    Returns a 0D tensor (scalar).
    """
    prob_clamped = prob.clamp(min=eps)
    return - (prob_clamped * prob_clamped.log2()).sum()


def expected_worker_differentiation_multiclass_vectorized(
    alpha: torch.Tensor,        # shape (K,)
    beta: torch.Tensor,         # shape (K,)
    worker_preds: torch.Tensor, # shape (N, K) => integer class predictions
    p_item: torch.Tensor,       # shape (N, C)
    item_idx: int,
    num_draws=1000,
    random_state=0,
    weighted=False,
    beta_update_weight=1.0
) -> float:
    """
    Vectorized multi-class Beta-Bernoulli approach:
      - current_prob_best = sample_is_best_worker(alpha, beta, ...)
      - alpha_batch,beta_batch = batch_update_beta_for_item(...)
      - prob_best_c = sample_is_best_worker_beta_batched(...)
      - ...
    Returns a float for the expected difference in "best worker" distribution.
    """
    # 1) Current distribution
    current_prob_best = sample_is_best_worker(alpha, beta, num_draws=num_draws, random_state=random_state)
    
    # 2) Build alpha_batch,beta_batch => shape (C,K)
    N, C_ = p_item.shape
    p_vec = p_item[item_idx]
    alpha_batch, beta_batch = batch_update_beta_for_item(
        alpha, beta,
        worker_preds[item_idx],
        C=C_,
        update_weight=beta_update_weight
    )

    # 3) single pass => (C, K)
    prob_best_c = sample_is_best_worker_beta_batched(
        alpha_batch, beta_batch, 
        # num_draws=num_draws,
        # random_state=random_state+999
    )

    # 4) Weighted L1 difference
    dist_sum = 0.0
    for c in range(C_):
        pc_val = p_vec[c].item()
        if pc_val < 1e-12:
            continue
        
        if weighted:
            # Weighted by current_prob_best
            changes = torch.abs(prob_best_c[c] - current_prob_best)
            diff_c = torch.dot(current_prob_best, changes)
        else:
            diff_c = torch.sum(torch.abs(prob_best_c[c] - current_prob_best)).item()
        
        dist_sum += pc_val * diff_c

    return dist_sum


def expected_information_gain_multiclass_vectorized(
    alpha: torch.Tensor,      # shape (K,)
    beta:  torch.Tensor,      # shape (K,)
    worker_preds: torch.Tensor, # shape (N, K)
    p_item: torch.Tensor,     # shape (N, C) => p_item[i,c]
    item_idx: int,
    num_draws=1000,
    random_state=0,
    beta_update_weight=1.0
) -> float:
    """
    Vectorized multi-class Beta-Bernoulli approach, measuring
    *expected reduction in entropy* of the "best worker" distribution.
    """
    current_prob_best = sample_is_best_worker(alpha, beta, num_draws=num_draws, random_state=random_state)
    H_current = distribution_entropy(current_prob_best)

    _, C_ = p_item.shape
    p_vec = p_item[item_idx]
    alpha_batch, beta_batch = batch_update_beta_for_item(
        alpha, beta,
        worker_preds[item_idx],
        C_,
        update_weight=beta_update_weight
    )

    prob_best_c = sample_is_best_worker_beta_batched(
        alpha_batch, beta_batch,
        # num_draws=num_draws,
        # random_state=random_state+999
    )

    dist_sum = 0.0
    for c in range(C_):
        pc_val = p_vec[c].item()
        if pc_val < 1e-12:
            continue
        H_c = distribution_entropy(prob_best_c[c])
        info_gain_c = (H_current - H_c).item()
        dist_sum += pc_val * info_gain_c

    return dist_sum


def compute_entropy(
    prob_2d: torch.Tensor,    # shape (X, K) - each row is a probability distribution
    weighted: bool = False,
    weights: torch.Tensor = None,  # shape(K,)
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Returns a 1D tensor of shape (X,) with standard or 'weighted' entropy.
    """
    if not weighted or weights is None:
        p_clamped = prob_2d.clamp_min(eps)
        ent = - (p_clamped * p_clamped.log2()).sum(dim=1)
        return ent

    # Weighted entropy
    w_2d = prob_2d * weights.unsqueeze(0)  # shape (X,K)
    sums = w_2d.sum(dim=1, keepdim=True).clamp_min(eps)
    w_dist_2d = w_2d / sums
    w_dist_clamped = w_dist_2d.clamp_min(eps)
    w_ent = - (w_dist_clamped * w_dist_clamped.log2()).sum(dim=1)
    return w_ent


def eig_batched_scatter(
    alpha: torch.Tensor,       
    beta:  torch.Tensor,        
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    candidate_ids,              
    chunk_size: int = 100,      # Increased chunk size possible
    weighted: bool = False,  
    eps: float = 1e-12,
    beta_update_weight=1.0
):
    device = alpha.device
    # candidate_ids = ensure_tensor(candidate_ids, device)
    candidate_ids = torch.tensor(candidate_ids, device=device)
    M = candidate_ids.shape[0]
    N, C_ = p_item.shape
    K_ = alpha.shape[0]

    # Precompute current distribution and entropy once
    current_prob_best = sample_is_best_worker(alpha, beta)
    if weighted:
        H_current = -(current_prob_best * current_prob_best.log2()).sum()
    else:
        H_current = -(current_prob_best.clamp_min(eps) * current_prob_best.clamp_min(eps).log2()).sum()

    # Precompute all p_item values in advance
    p_item_vals = p_item[candidate_ids]  # (M, C)

    # Batch process probabilities for all candidates first
    all_probs = []
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        batch_ids = candidate_ids[start:end]
        
        alpha_chunk, beta_chunk = build_alpha_beta_chunk(
            alpha, beta, worker_preds, batch_ids, C_,
            update_weight=beta_update_weight
        )
        
        probs = sample_is_best_worker_beta_batched(alpha_chunk, beta_chunk)  # (bs, C, K)
        all_probs.append(probs)
    
    # Concatenate all probabilities (M, C, K)
    all_probs = torch.cat(all_probs, dim=0)

    # Vectorized entropy calculation
    if weighted:
        weights = current_prob_best.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        entropies = compute_batched_weighted_entropy(all_probs, weights, eps)
    else:
        p_clamped = all_probs.clamp_min(eps)
        entropies = -(p_clamped * p_clamped.log2()).sum(dim=-1)  # (M, C)

    # Vectorized EIG calculation
    H_diff = H_current - entropies  # (M, C)
    EIG_vals = (p_item_vals * H_diff).sum(dim=-1)

    return EIG_vals

def compute_batched_weighted_entropy(probs, weights, eps):
    """Vectorized computation of weighted entropy for (N, C, K) tensor"""
    p_clamped = probs.clamp_min(eps)
    weighted_terms = weights * p_clamped * torch.log(p_clamped)
    return -weighted_terms.sum(dim=-1)  # (N, C)

def build_alpha_beta_chunk(alpha, beta, worker_preds, batch_ids, C, update_weight=1.0):
    device = alpha.device
    K_ = alpha.shape[0]
    M_ = batch_ids.shape[0]

    alpha_chunk = alpha.view(1,1,K_).expand(M_, C, K_).clone()
    beta_chunk  = beta.view(1,1,K_).expand(M_, C, K_).clone()

    sub_preds = worker_preds[batch_ids]
    for c in range(C):
        eq_mask = (sub_preds == c)
        alpha_chunk[:, c, :][eq_mask] += 1.0 * update_weight
        beta_chunk[:, c, :][~eq_mask] += 1.0 * update_weight

    return alpha_chunk, beta_chunk


def weighted_disagreement_candidates(preds, prob_best, k=5, n=50):
    """
    Same as original code, unchanged.
    """
    top_probs, topk_idxs = torch.topk(prob_best, k=k)
    preds_top = preds[:, topk_idxs]
    majority_class, _ = torch.mode(preds_top, dim=1)
    not_majority_mask = (preds_top != majority_class.unsqueeze(1))
    scores = not_majority_mask.float() * top_probs.unsqueeze(0)
    scores = scores.sum(dim=1)
    _, topn_indices = torch.topk(scores, n)
    return topn_indices


def expected_regret_multiclass_vectorized(
    alpha: torch.Tensor,          
    beta:  torch.Tensor,          
    worker_preds: torch.Tensor,   # (N, K)
    p_item: torch.Tensor,         # (N, C)
    item_idx: int,
    num_draws=1000,
    random_state=0,
    loss_fn=None
) -> float:
    """
    Same as original, calls sample_is_best_worker(...) to get prob_best.
    """
    device = alpha.device
    prob_best = sample_is_best_worker(alpha, beta, num_draws=num_draws, random_state=random_state)
    best_model_idx = torch.argmax(prob_best).item()
    map_pred = worker_preds[item_idx, best_model_idx]

    if loss_fn is None:
        pass

    model_preds_for_item = worker_preds[item_idx]
    p_vec = p_item[item_idx]
    C_ = p_vec.shape[0]

    classes = torch.arange(C_, device=device)
    loss_map = (map_pred.unsqueeze(0) != classes).float()
    preds_eq_c = model_preds_for_item.unsqueeze(0) == classes.unsqueeze(1)
    zero_one_loss_mat = (~preds_eq_c).float()
    mixture_per_class = zero_one_loss_mat * prob_best.unsqueeze(0)
    loss_mix = mixture_per_class.sum(dim=1)
    diff_per_class = loss_map - loss_mix
    expected_regret_val = torch.dot(p_vec, diff_per_class).item()
    return expected_regret_val


def local_regret_one_item(
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    item_idx: int,
    prob_best_1d: torch.Tensor,  # shape(K,)
    map_idx: int,
    loss_fn
) -> float:
    """
    Unchanged from original code.
    """
    device = worker_preds.device
    C_ = p_item.shape[1]
    map_pred = worker_preds[item_idx, map_idx].item()

    local_loss_map = torch.zeros(C_, dtype=torch.float, device=device)
    mixture_loss   = torch.zeros(C_, dtype=torch.float, device=device)
    K_ = prob_best_1d.shape[0]
    for c in range(C_):
        pc_val = p_item[item_idx, c].item()
        if pc_val < 1e-12:
            continue
        local_loss_map[c] = float(loss_fn(map_pred, c))
        sum_loss = 0.0
        for h_idx in range(K_):
            h_pred = worker_preds[item_idx, h_idx].item()
            sum_loss += prob_best_1d[h_idx].item() * float(loss_fn(h_pred, c))
        mixture_loss[c] = sum_loss

    diff_per_class = local_loss_map - mixture_loss
    local_reg_val = torch.dot(p_item[item_idx], diff_per_class).item()
    return local_reg_val


def largest_regret_reduction_global(
    alpha: torch.Tensor, 
    beta: torch.Tensor,
    worker_preds: torch.Tensor,   # shape(N,K)
    p_item: torch.Tensor,         # shape(N,C)
    unlabeled_indices: torch.Tensor, 
    item_idx: int,
    num_draws=1000,
    random_state=0,
    loss_fn=None
) -> float:
    """
    Unchanged from original code.
    """
    device = alpha.device
    if loss_fn is None:
        def loss_fn(pred_label, true_label):
            return 1.0 if (pred_label != true_label) else 0.0

    old_global_val = compute_global_regret(
        alpha, beta,
        worker_preds, p_item, unlabeled_indices,
        num_draws, random_state, loss_fn
    )

    N, C_ = p_item.shape
    alpha_batch, beta_batch = batch_update_beta_for_item(
        alpha, beta, 
        worker_preds[item_idx], 
        C=C_
    )
    prob_best_c = sample_is_best_worker_beta_batched(
        alpha_batch, beta_batch,
        # num_draws=num_draws,
        # random_state=(random_state+999)
    )

    new_unlabeled = []
    item_idx_int = int(item_idx)
    for x in unlabeled_indices:
        xx = int(x.item()) if hasattr(x, "item") else int(x)
        if xx != item_idx_int:
            new_unlabeled.append(xx)
    new_unlabeled = torch.tensor(new_unlabeled, dtype=torch.long, device=device)

    new_global_sum = 0.0
    for c in range(C_):
        pc_val = p_item[item_idx,c].item()
        if pc_val < 1e-12:
            continue
        new_global_c = compute_global_regret_after_scenario(
            new_unlabeled,
            worker_preds, p_item,
            prob_best_c[c],
            alpha_batch[c],
            beta_batch[c],
            loss_fn
        )
        new_global_sum += pc_val * new_global_c

    new_global_expected = new_global_sum
    return old_global_val - new_global_expected


def compute_global_regret(
    alpha, beta,
    worker_preds, p_item,
    unlabeled_indices,
    num_draws=1000,
    random_state=0,
    loss_fn=None
):
    """
    Unchanged from original code.
    """
    if loss_fn is None:
        def loss_fn(a,b): return 1.0 if a != b else 0.0

    prob_best = sample_is_best_worker(alpha,beta,num_draws,random_state)
    map_idx = torch.argmax(prob_best).item()

    total = 0.0
    for j_ in unlabeled_indices:
        j = int(j_.item()) if hasattr(j_,"item") else int(j_)
        local_val = local_regret_one_item(
            worker_preds, p_item, j,
            prob_best, map_idx, loss_fn
        )
        total += local_val
    return total


def compute_global_regret_after_scenario(
    new_unlabeled,
    worker_preds, p_item,
    prob_best_c,
    alpha_c, beta_c,
    loss_fn
):
    """
    Unchanged from original code.
    """
    if loss_fn is None:
        def loss_fn(a,b): return 1.0 if a!=b else 0.0

    map_idx_c = torch.argmax(prob_best_c).item()

    total = 0.0
    for j_ in new_unlabeled:
        j = int(j_.item()) if hasattr(j_,"item") else int(j_)
        local_val = local_regret_one_item(
            worker_preds, p_item, j,
            prob_best_c, map_idx_c, loss_fn
        )
        total += local_val
    return total


def largest_regret_reduction_global_fast(
    alpha: torch.Tensor, 
    beta: torch.Tensor,
    worker_preds: torch.Tensor,    # shape (N,K)
    p_item: torch.Tensor,          # shape (N,C)
    unlabeled_indices: torch.Tensor,
    item_idx: int,
    num_draws=1000,
    random_state=0,
    beta_update_weight=1.0
) -> float:
    """
    Unchanged from original code.
    """
    device = alpha.device
    old_global_val = compute_global_regret_fast(
        alpha, beta,
        worker_preds, p_item,
        unlabeled_indices,
        num_draws=num_draws,
        random_state=random_state
    )

    N, C_ = p_item.shape
    alpha_batch, beta_batch = batch_update_beta_for_item(
        alpha, beta, 
        worker_preds[item_idx],
        C=C_,
        update_weight=beta_update_weight
    )
    prob_best_c = sample_is_best_worker_beta_batched(
        alpha_batch, beta_batch,
        num_draws=num_draws,
        random_state=(random_state+999)
    )

    new_unlabeled = []
    item_idx_int = int(item_idx)
    for x_ in unlabeled_indices:
        x_int = int(x_.item()) if hasattr(x_,"item") else int(x_)
        if x_int != item_idx_int:
            new_unlabeled.append(x_int)
    new_unlabeled = torch.tensor(new_unlabeled, dtype=torch.long, device=device)

    new_global_sum = 0.0
    for c in range(C_):
        pc_val = p_item[item_idx,c].item()
        if pc_val < 1e-12:
            continue
        new_global_c = compute_global_regret_fast_for_scenario(
            new_unlabeled, worker_preds, p_item,
            prob_best_c[c],
        )
        new_global_sum += pc_val * new_global_c

    new_global_expected = new_global_sum
    return old_global_val - new_global_expected


def compute_global_regret_fast(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    worker_preds: torch.Tensor, 
    p_item: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    num_draws=1000,
    random_state=0,
    loss_fn=None
) -> float:
    """
    Unchanged from original code.
    """
    device = alpha.device
    if loss_fn is None:
        def loss_fn(a,b): return 1.0 if a != b else 0.0

    prob_best = sample_is_best_worker(alpha,beta,num_draws,random_state)
    map_idx = torch.argmax(prob_best).item()

    return compute_local_regret_sum(
        worker_preds, p_item,
        unlabeled_indices, prob_best, map_idx
    )


def compute_global_regret_fast_for_scenario(
    unlabeled_indices: torch.Tensor,
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    prob_best_1d: torch.Tensor,
    loss_fn=None
) -> float:
    """
    Unchanged from original code.
    """
    if loss_fn is None:
        def loss_fn(a,b): return 1.0 if a != b else 0.0

    map_idx = torch.argmax(prob_best_1d).item()
    return compute_local_regret_sum(
        worker_preds, p_item,
        unlabeled_indices, prob_best_1d, map_idx
    )


def compute_local_regret_sum(
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    prob_best_1d: torch.Tensor,
    map_idx: int
) -> float:
    """
    Unchanged from original code.
    """
    device = worker_preds.device
    sub_preds = worker_preds[unlabeled_indices]
    sub_p     = p_item[unlabeled_indices]
    M = sub_preds.shape[0]
    C_ = sub_p.shape[1]

    map_preds_m = sub_preds[:, map_idx]
    classes = torch.arange(C_, device=device)
    map_loss_mC = (map_preds_m.unsqueeze(1) != classes.unsqueeze(0)).float()

    pred_diff_mhC = (sub_preds.unsqueeze(2) != classes.view(1,1,C_)).float()
    mixture_mC = (pred_diff_mhC * prob_best_1d.view(1,-1,1)).sum(dim=1)
    local_regret_mC = map_loss_mC - mixture_mC
    local_regret_m  = (local_regret_mC * sub_p).sum(dim=1)
    return local_regret_m.sum().item()


def largest_regret_reduction_global_fast_final(
    alpha: torch.Tensor, 
    beta: torch.Tensor,
    worker_preds: torch.Tensor,    # shape (N,K)
    p_item: torch.Tensor,          # shape (N,C)
    unlabeled_indices,
    item_idx: int,
    num_draws=1000,
    random_state=0,
    beta_update_weight=1.0
) -> float:
    """
    Unchanged from original code.
    """
    device = alpha.device
    if not isinstance(unlabeled_indices, torch.Tensor):
        unlabeled_indices = torch.tensor(unlabeled_indices, dtype=torch.long, device=device)

    N, C_ = p_item.shape

    if not (unlabeled_indices == item_idx).any():
        print(f"[Warning] item_idx={item_idx} not found in unlabeled_indices => can't reduce regret. Returning -9999.")
        return -9999.0

    old_global_val = compute_global_regret_fast_fixed(
        alpha, beta,
        worker_preds, p_item,
        unlabeled_indices,
        num_draws=num_draws,
        random_state=random_state
    )

    alpha_batch, beta_batch = batch_update_beta_for_item(
        alpha, beta,
        worker_preds[item_idx],
        C=C_,
        update_weight=beta_update_weight
    )
    prob_best_c = sample_is_best_worker_beta_batched(
        alpha_batch, beta_batch,
        # num_draws=num_draws,
        # random_state=random_state+999
    )

    keep_mask = (unlabeled_indices != item_idx)
    new_unlabeled = unlabeled_indices[keep_mask]

    new_global_sum = 0.0
    for c in range(C_):
        pc_val = p_item[item_idx, c].item()
        if pc_val < 1e-12:
            continue
        scenario_val = compute_global_regret_fast_scenario_fixed(
            new_unlabeled,
            worker_preds, p_item,
            prob_best_c[c]
        )
        new_global_sum += pc_val * scenario_val

    new_global_expected = new_global_sum
    return old_global_val - new_global_expected


def compute_global_regret_fast_fixed(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    num_draws=1000,
    random_state=0,
    loss_fn=None
) -> float:
    """
    Unchanged from original code.
    """
    if loss_fn is None:
        def loss_fn(a,b): return 1.0 if a!=b else 0.0

    prob_best = sample_is_best_worker(alpha, beta)
    map_idx = torch.argmax(prob_best).item()
    return sum_local_regrets_fast(
        worker_preds, p_item,
        unlabeled_indices,
        prob_best, map_idx
    )


def compute_global_regret_fast_scenario_fixed(
    new_unlabeled: torch.Tensor,
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    prob_best_1d: torch.Tensor,
    loss_fn=None
) -> float:
    """
    Unchanged from original code.
    """
    if loss_fn is None:
        def loss_fn(a,b): return 1.0 if a!=b else 0.0
    map_idx = torch.argmax(prob_best_1d).item()
    return sum_local_regrets_fast(
        worker_preds, p_item,
        new_unlabeled,
        prob_best_1d, map_idx
    )


def sum_local_regrets_fast(
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    prob_best_1d: torch.Tensor,
    map_idx: int
) -> float:
    """
    Unchanged from original code.
    """
    device = worker_preds.device
    sub_preds = worker_preds[unlabeled_indices]
    sub_p     = p_item[unlabeled_indices]
    M = sub_preds.shape[0]
    C_ = sub_p.shape[1]

    map_preds_m = sub_preds[:, map_idx]
    classes = torch.arange(C_, device=device)
    map_loss_mC = (map_preds_m.unsqueeze(1) != classes.unsqueeze(0)).float()

    pred_diff_mhC = (sub_preds.unsqueeze(2) != classes.view(1,1,C_)).float()
    mixture_mC = (pred_diff_mhC * prob_best_1d.view(1,-1,1)).sum(dim=1)
    local_regret_mC = map_loss_mC - mixture_mC
    local_regret_m  = (local_regret_mC * sub_p).sum(dim=1)
    return local_regret_m.sum().item()


def beta_update(a: torch.Tensor,
                b: torch.Tensor,
                outcome: torch.Tensor,
                weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conjugate Beta update for a single item (n=1 per model),
    with optional importance weight 'weight'.
    """
    a_new = a + weight * outcome
    b_new = b + weight * (1. - outcome)
    return a_new, b_new


def expected_worker_differentiation_for_items_batch(
    alpha, beta,
    worker_preds,  # (N,K)
    p_item,        # (N,C)
    candidate_items,  # list/1D-tensor
    num_draws=1000,
    random_state=0,
    weighted=False
) -> torch.Tensor:
    """
    Returns a 1D tensor `diff_values` of shape(len(candidate_items)),
    each is the 'expected_worker_differentiation' for that item,
    calling sample_is_best_worker(...) and sample_is_best_worker_beta_batched(...)
    for each row.  Now those use integrals under the hood.
    """
    device = alpha.device
    if not isinstance(candidate_items, torch.Tensor):
        candidate_items = torch.tensor(candidate_items, device=device, dtype=torch.long)
    M = candidate_items.shape[0]
    N, C_ = p_item.shape
    K_ = alpha.shape[0]

    current_prob_best = sample_is_best_worker(alpha, beta, num_draws, random_state)

    alpha_big = alpha.view(1,1,K_).expand(M, C_, K_).clone()
    beta_big  = beta.view(1,1,K_).expand(M, C_, K_).clone()

    sub_preds = worker_preds[candidate_items]
    for c in range(C_):
        eq_mask = (sub_preds == c)
        alpha_big[:, c, :][eq_mask] += 1.0
        beta_big[:, c, :][~eq_mask] += 1.0

    alpha_flat = alpha_big.view(-1)
    beta_flat  = beta_big.view(-1)

    # We'll do a single for-loop in Python to do the integral row by row
    # for each m in [0..M-1], we produce shape (C,K). Then compute L1 vs current_prob_best.
    diff_values = torch.zeros(M, device=device)
    for m in tqdm(range(M)):
        alpha_m = alpha_big[m]  # shape (C,K)
        beta_m  = beta_big[m]   # shape (C,K)
        prob_best_c = sample_is_best_worker_beta_batched(alpha_m, 
                                                         beta_m,
                                                        #  num_points=64
                                                         )

        diffs_CK = (prob_best_c - current_prob_best.view(1,K_)).abs()  # shape(C,K)
        if weighted:
            # Weighted by current_prob_best
            diffs_C = (diffs_CK * current_prob_best.view(1,K_)).sum(dim=1)  # shape(C,)
        else:
            diffs_C = diffs_CK.sum(dim=1)  # shape(C,)

        i_m = candidate_items[m].item()
        p_vec = p_item[i_m]
        diff_values[m] = (p_vec * diffs_C).sum()

    return diff_values


def worker_diff_batched_scatter(
    alpha: torch.Tensor,       
    beta: torch.Tensor,        
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    candidate_ids,
    chunk_size: int = 50,
    weighted: bool = False,
    beta_update_weight: float = 1.0,
    eps: float = 1e-12
):
    """
    Batched version of expected_worker_differentiation_multiclass_vectorized
    """
    device = alpha.device
    if not isinstance(candidate_ids, torch.Tensor):
        candidate_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        
    M = candidate_ids.shape[0]
    N, C_ = p_item.shape
    K_ = alpha.shape[0]
    
    # Current best worker distribution
    current_prob_best = sample_is_best_worker(alpha, beta)
    results = torch.zeros(M, device=device)

    # Process in chunks
    start = 0
    while start < M:
        end = min(start + chunk_size, M)
        batch_ids = candidate_ids[start:end]
        batch_size = end - start

        # Build batched alpha/beta parameters for all items in chunk
        alpha_chunk, beta_chunk = build_alpha_beta_chunk(
            alpha, beta, worker_preds, batch_ids, C_,
            update_weight=beta_update_weight
        )  # shapes (batch_size, C, K)

        # Batch compute probability distributions
        prob_best_chunk = sample_is_best_worker_beta_batched(
            alpha_chunk, beta_chunk
        )  # (batch_size, C, K)

        # Vectorized difference computation
        current_expanded = current_prob_best.unsqueeze(0).unsqueeze(1)  # (1, 1, K)
        differences = torch.abs(prob_best_chunk - current_expanded)
        
        if weighted:
            weights = current_prob_best.unsqueeze(0).unsqueeze(1)  # (1, 1, K)
            diffs = (differences * weights).sum(dim=-1)  # (batch_size, C)
        else:
            diffs = differences.sum(dim=-1)  # (batch_size, C)

        # Weight by p_item and sum over classes
        p_values = p_item[batch_ids]  # (batch_size, C)
        batch_results = (p_values * diffs).sum(dim=-1)
        
        results[start:end] = batch_results
        start = end

    return results

# def regret_reduction_batched_scatter(
#     alpha: torch.Tensor,
#     beta: torch.Tensor,
#     worker_preds: torch.Tensor,
#     p_item: torch.Tensor,
#     unlabeled_indices,
#     candidate_ids,
#     chunk_size: int = 20,
#     beta_update_weight: float = 1.0,
#     num_draws: int = 500
# ):
#     device = alpha.device
#     if not isinstance(candidate_ids, torch.Tensor):
#         candidate_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    
#     # Convert unlabeled_indices to tensor if needed
#     if not isinstance(unlabeled_indices, torch.Tensor):
#         unlabeled_indices = torch.tensor(unlabeled_indices, dtype=torch.long, device=device)
    
#     M = candidate_ids.shape[0]
#     N, C_ = p_item.shape
    
#     # Precompute old global value
#     old_global_val = compute_global_regret_fast_fixed(
#         alpha, beta, worker_preds, p_item, unlabeled_indices, num_draws
#     )
    
#     results = torch.zeros(M, device=device)
    
#     # Create keep masks using vectorized operations
#     keep_masks = unlabeled_indices != candidate_ids.unsqueeze(-1)  # (M, num_unlabeled)
    
#     start = 0
#     while start < M:
#         end = min(start + chunk_size, M)
#         batch_ids = candidate_ids[start:end]
#         batch_size = end - start
        
#         # Build batched parameters
#         alpha_chunk, beta_chunk = build_alpha_beta_chunk(
#             alpha, beta, worker_preds, batch_ids, C_,
#             # update_weight=beta_update_weight
#         )
        
#         # Batch compute probability distributions
#         prob_best_chunk = sample_is_best_worker_beta_batched(alpha_chunk, beta_chunk)
        
#         # Batch process scenarios
#         batch_masks = keep_masks[start:end]  # (batch_size, num_unlabeled)
#         p_values = p_item[batch_ids]  # (batch_size, C)
        
#         # Vectorized computation
#         batch_reductions = compute_batched_global_regret(
#             prob_best_chunk, 
#             p_values, 
#             batch_masks,
#             worker_preds,
#             p_item,
#             unlabeled_indices
#         )
        
#         results[start:end] = old_global_val - batch_reductions
#         start = end

#     return results

# def regret_reduction_batched_scatter(
#     alpha: torch.Tensor,
#     beta: torch.Tensor,
#     worker_preds: torch.Tensor,
#     p_item: torch.Tensor,
#     unlabeled_indices: torch.Tensor,
#     candidate_ids: torch.Tensor,
#     chunk_size: int = 50,
#     beta_update_weight: float = 1.0
# ) -> torch.Tensor:
    
#     device = alpha.device
#     candidate_ids = torch.tensor(candidate_ids, device=device)
#     unlabeled_indices = torch.tensor(unlabeled_indices, device=device)
#     N, C = p_item.shape
#     U = unlabeled_indices.shape[0]
#     K = alpha.shape[0]

#     # Precompute baseline global regret
#     old_global = compute_global_regret_fast_fixed(
#         alpha, beta, worker_preds, p_item, unlabeled_indices
#     )

#     # Precompute all worker predictions for unlabeled items
#     wp_unlabeled = worker_preds[unlabeled_indices]  # (U, K)
    
#     # Process in chunks
#     results = torch.zeros(len(candidate_ids), device=device)
    
#     for chunk_start in range(0, len(candidate_ids), chunk_size):
#         chunk_end = min(chunk_start + chunk_size, len(candidate_ids))
#         chunk_ids = candidate_ids[chunk_start:chunk_end]
#         chunk_size_actual = chunk_end - chunk_start
        
#         # 1. Batch update parameters (chunk_size, C, K)
#         alpha_chunk, beta_chunk = build_alpha_beta_chunk(
#             alpha, beta, worker_preds, chunk_ids, C, 
#             # beta_update_weight
#         )
#         probs = sample_is_best_worker_beta_batched(alpha_chunk, beta_chunk)

#         # 2. Prepare indices for advanced indexing
#         index_c = wp_unlabeled  # (U, K)
#         index_k = torch.arange(K, device=device)[None,:].expand(U, K)  # (U, K)

#         # 3. Gather probabilities using vectorized indexing (chunk_size, U, K)
#         gathered_probs = probs[:, index_c, index_k]  # Correct class dimension indexing

#         # 4. Create exclusion masks (chunk_size, U)
#         keep_masks = unlabeled_indices != chunk_ids.view(-1,1)
        
#         # 5. Apply masks and sum
#         masked_probs = gathered_probs * keep_masks.unsqueeze(-1).float()
#         sum_probs = masked_probs.sum(dim=(1,2))  # (chunk_size,)

#         # 6. Compute final regret reduction
#         results[chunk_start:chunk_end] = old_global - (1 - sum_probs)

#     return results

def regret_reduction_batched_scatter(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    candidate_ids: torch.Tensor,
    chunk_size: int = 50,
    beta_update_weight: float = 1.0
) -> torch.Tensor:
    """
    Vectorized version that matches the logic of largest_regret_reduction_global_fast_final
    for multiple candidates in one pass, removing explicit per-item/per-class loops.

    Returns:
        results: Tensor of shape (len(candidate_ids),) giving the regret reduction for each candidate.
    """

    device = alpha.device
    candidate_ids = torch.tensor(candidate_ids, device=device)
    unlabeled_indices = torch.tensor(unlabeled_indices, device=device)

    # Basic shapes
    N, C = p_item.shape
    K = alpha.shape[0]
    U = len(unlabeled_indices)
    num_candidates = len(candidate_ids)

    # 1) Compute baseline global regret (same for all)
    old_global = compute_global_regret_fast_fixed(
        alpha, beta, worker_preds, p_item, unlabeled_indices
    )

    # 2) Precompute "error_for_worker[n, k]" = expected error if worker k is chosen for item n
    #    error_for_worker[n, k] = 1 - p_item[n, worker_preds[n,k]]
    #    shape = (N, K).
    row_idx = torch.arange(N, device=device).unsqueeze(1)     # shape (N,1)
    col_idx = worker_preds                                   # shape (N,K)
    p_correct_for_worker = p_item[row_idx, col_idx]          # shape (N,K)
    error_for_worker = 1.0 - p_correct_for_worker            # shape (N,K)

    # 3) We'll write our final results here
    results = torch.zeros(num_candidates, device=device)

    # 4) Chunk over the candidate set to limit memory usage
    for chunk_start in range(0, num_candidates, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_candidates)
        chunk_ids = candidate_ids[chunk_start:chunk_end]   # shape (chunk_size,)
        size_here = chunk_end - chunk_start

        # (a) Build alpha, beta for these items => shape (size_here, C, K)
        alpha_chunk, beta_chunk = build_alpha_beta_chunk(
            alpha, beta, worker_preds, chunk_ids, C,
            update_weight=beta_update_weight
        )
        # (b) Probability distribution over which worker is "best" if item i has class c
        #     shape = (size_here, C, K)
        probs_chunk = sample_is_best_worker_beta_batched(alpha_chunk, beta_chunk)

        # (c) Build a mask that excludes each candidate from the unlabeled set:
        #     mask[i, u] = 1 if unlabeled_indices[u] != chunk_ids[i], else 0
        #     shape = (size_here, U)
        candidate_ids_view = chunk_ids.view(size_here, 1)   # shape (size_here,1)
        mask = (unlabeled_indices.unsqueeze(0) != candidate_ids_view).float()  # (size_here,U)

        # (d) Now compute scenario_val[i, c] = sum_{u,k} mask[i,u] * error_for_worker[unlabeled_indices[u], k] * probs_chunk[i,c,k].
        #
        #     We'll use broadcasting so that the final summation is a .sum(dim=(2,3)).
        #
        #     - mask           => shape (size_here, U) -> view as (size_here, 1, U, 1)
        #     - error_for_worker[...] => shape (U, K) after indexing unlabeled_indices -> view as (1, 1, U, K)
        #     - probs_chunk    => shape (size_here, C, K) -> view as (size_here, C, 1, K)

        mask_4d  = mask[:, None, :, None]  # (size_here, 1, U, 1)
        err_4d   = error_for_worker[unlabeled_indices]      # (U, K)
        err_4d   = err_4d.unsqueeze(0).unsqueeze(0)         # (1, 1, U, K)
        probs_4d = probs_chunk.unsqueeze(2)                 # (size_here, C, 1, K)

        # elementwise multiply => shape (size_here, C, U, K)
        scenario_tensor = mask_4d * err_4d * probs_4d
        # sum over U,K => scenario_val[i, c]
        scenario_val = scenario_tensor.sum(dim=(2,3))  # shape (size_here, C)

        # (e) For each candidate i in the chunk, we weight scenario_val[i,c] by p_item[candidate, c]
        #     then sum over c => expected new global regret if that candidate is labeled
        #     exactly as in the original code:
        #
        #     new_global[i] = sum_c p_item[cand, c] * scenario_val[i,c]
        #
        # shape of p_cand = (size_here, C)
        p_cand = p_item[chunk_ids, :]  # gather the p_item row for each candidate
        new_global = (p_cand * scenario_val).sum(dim=1)  # shape (size_here,)

        # (f) The final regret reduction for each candidate = old_global - new_global
        results[chunk_start:chunk_end] = old_global - new_global

    return results



# Helper function needed for regret computation
def compute_batched_global_regret(
    prob_best_chunk: torch.Tensor,  # (batch_size, C, K)
    p_values: torch.Tensor,         # (batch_size, C)
    keep_masks: torch.Tensor,       # (batch_size, num_unlabeled)
    worker_preds: torch.Tensor,
    p_item_full: torch.Tensor,
    unlabeled_indices: torch.Tensor
):
    device = prob_best_chunk.device
    batch_size, C_, K_ = prob_best_chunk.shape
    num_unlabeled = unlabeled_indices.shape[0]
    
    # Get worker predictions for unlabeled items => (num_unlabeled, K)
    worker_preds_unlabeled = worker_preds[unlabeled_indices]
    
    # Prepare indices for gather operation (class dimension)
    indices = worker_preds_unlabeled.unsqueeze(0)  # (1, num_unlabeled, K)
    indices = indices.expand(batch_size, -1, -1).long()  # (batch_size, num_unlabeled, K)
    
    # Gather probabilities along CLASS dimension (dim=1)
    gathered_probs = prob_best_chunk.gather(
        1,  # Dimension to gather from (C)
        indices  # Shape (batch_size, num_unlabeled, K)
    )  # Output shape (batch_size, num_unlabeled, K)
    
    # Apply keep masks and sum
    masked_probs = gathered_probs * keep_masks.unsqueeze(-1).float()  # (batch_size, num_unlabeled, K)
    summed_probs = masked_probs.sum(dim=1)  # (batch_size, K)
    
    # Compute regret terms
    regret_terms = 1.0 - summed_probs  # (batch_size, K)
    
    # Weight by class probabilities and aggregate
    weighted_regret = p_values.unsqueeze(-1) * regret_terms.unsqueeze(1)  # (batch_size, C, K)
    return weighted_regret.sum(dim=[1,2])  # (batch_size,)


class BB(ModelSelector):
    """
    Beta-Bernoulli conjugate approach with priors from some surrogate (e.g. ensemble or DS).
    """
    def __init__(self, dataset, 
                 prior_source="ens-exp",        # how to set prior accuracies
                 prior_strength=10.0,           # scale of initial alpha, beta 
                 q='eig',                       # acquisition function
                 select='sample',               # how to compute P(h=h*)
                 stochastic=False,              # sample according to q if true; greedy if false
                 importance_weighting=False,    # if sampling, also use importance weights
                 prefilter_fn='disagreement',   # filter with heuristic
                 prefilter_n=0,                 # number to filter down to
                 epsilon=0.0,                   
                 item_prior_source="ens",
                 update_strength=1.0
                 ):
        self.dataset = dataset
        self.device = dataset.pred_logits.device
        self.H, self.N, self.C = dataset.pred_logits.shape
        
        ensemble = Ensemble(dataset.pred_logits)
        ensemble_preds = ensemble.get_preds() # (N, C)
        one_hot_preds = torch.nn.functional.one_hot(torch.argmax(dataset.pred_logits, dim=-1), num_classes=self.C).float().mean(dim=0)

        if prior_source == "ens-exp":
            self.pred_losses = metrics.simple_expected_error(ensemble_preds, dataset.pred_logits).mean(dim=-1)
        elif prior_source == "ens-01":
            self.pred_losses = metrics.simple_error(one_hot_preds, dataset.pred_logits)
        elif prior_source == "ens-soft-01":
            self.pred_losses = metrics.simple_error(ensemble_preds, dataset.pred_logits)
        elif prior_source == "ds":
            dataset_tensor = torch.nn.functional.softmax(dataset.pred_logits.permute(1,0,2), dim=-1)
            model = DawidSkeneModel(self.C, max_iter=100, tolerance=1e-10)
            _, _, worker_reliability, _ = model.run(dataset_tensor)
            self.pred_losses = 1 - worker_reliability

        accuracy_priors = 1 - self.pred_losses
        self.alpha = 1 + accuracy_priors * prior_strength
        self.beta = 1 + (1 - accuracy_priors) * prior_strength

        self.item_prior_source = item_prior_source
        if item_prior_source == "ens":
            self.item_priors = ensemble_preds
        elif item_prior_source == "ens-01":
            self.item_priors = one_hot_preds
        elif item_prior_source == "none" or item_prior_source == "uniform":
            self.item_priors = torch.ones((self.N, self.C), device=self.device) * 1/self.C
        elif item_prior_source == "bma-adaptive":
            self.item_priors = self.get_bma_preds()
        else:
            raise NotImplemented
        
        self.q = q
        self.select = select
        self.stochastic = stochastic
        self.prefilter_fn = prefilter_fn
        self.prefilter_n = prefilter_n
        self.prefilter_h = 50
        self.importance_weighting = importance_weighting
        self.epsilon = epsilon
        self.update_strength = update_strength

        self.d_l_idxs = []
        self.d_l_ys = []
        self.d_u_idxs = list(range(dataset.pred_logits.shape[1]))
        self.qms = []

    def get_next_item_to_label(self):
        pred_classes = torch.argmax(self.dataset.pred_logits, dim=-1)
        # q_values = torch.zeros(self.N, device=self.device)

        if self.item_prior_source == 'bma-adaptive':
            self.item_priors = self.get_bma_preds()

        _d_u_idxs = self.d_u_idxs
        if self.prefilter_fn == 'disagreement':
            # candidate_list = list(weighted_disagreement_candidates(
            #     pred_classes.T[self.d_u_idxs],
            #     torch.tensor(prob_best, device=pred_classes.device), 
            #     k=min(self.prefilter_h, self.H),
            #     n=min(self.prefilter_n, self.N)
            # ))
            # _d_u_idxs = [ self.d_u_idxs[i] for i in candidate_list ]
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
            _q_values = eig_batched_scatter(
                self.alpha, self.beta,
                pred_classes.T,
                self.item_priors,
                candidate_ids=_d_u_idxs,
                beta_update_weight=self.update_strength
                # num_draws=500
            )
        elif q =='min_eig':
            _q_values = -1 * eig_batched_scatter(
                self.alpha, self.beta,
                pred_classes.T,
                self.item_priors,
                candidate_ids=_d_u_idxs,
                beta_update_weight=self.update_strength
                # num_draws=500
            )
        elif q== 'weighted_eig':
            _q_values = eig_batched_scatter(
                self.alpha, self.beta,
                pred_classes.T,
                self.item_priors,
                candidate_ids=_d_u_idxs,
                beta_update_weight=self.update_strength,
                # num_draws=500,
                weighted=True
            )
        elif q == 'l1':
            # q_values[_d_u_idxs] = expected_worker_differentiation_for_items_batch(
            _q_values = worker_diff_batched_scatter(
                self.alpha, self.beta,
                pred_classes.T,
                self.item_priors,
                candidate_ids=_d_u_idxs,
                weighted=False,
                beta_update_weight=self.update_strength
            )
        elif q == 'weighted_l1': # and self.C < 1000:
            _q_values = worker_diff_batched_scatter( #expected_worker_differentiation_for_items_batch(
                self.alpha, self.beta,
                pred_classes.T,
                self.item_priors,
                candidate_ids=_d_u_idxs,
                weighted=True,
                beta_update_weight=self.update_strength
            )
        else:
            _q_values = torch.zeros(len(_d_u_idxs), device=self.device)
            for i, j in tqdm(enumerate(_d_u_idxs), desc=f'computing {q}'):
                # if q == 'weighted_l1':
                #     l1_j = expected_worker_differentiation_multiclass_vectorized(
                #         self.alpha, self.beta,
                #         pred_classes.T,
                #         self.item_priors,
                #         item_idx=j,
                #         num_draws=500,
                #         weighted=True
                #     )
                #     q_values[j] = l1_j
                if q == 'max_regret':
                    reg_j = expected_regret_multiclass_vectorized(
                        self.alpha, self.beta,
                        pred_classes.T,
                        self.item_priors,
                        item_idx=j,
                        num_draws=500
                    )
                    _q_values[i] = reg_j
                elif q == 'reduce_regret_global':
                    reg_j = largest_regret_reduction_global_fast_final(
                        self.alpha, self.beta,
                        pred_classes.T,
                        self.item_priors,
                        _d_u_idxs,
                        item_idx=j,
                        num_draws=500,
                        beta_update_weight=self.update_strength
                    )
                    _q_values[i] = reg_j
                elif q == 'increase_regret_global':
                    reg_j = largest_regret_reduction_global_fast_final(
                        self.alpha, self.beta,
                        pred_classes.T,
                        self.item_priors,
                        _d_u_idxs,
                        item_idx=j,
                        num_draws=500,
                        beta_update_weight=self.update_strength
                    )
                    _q_values[i] = -1 * reg_j
                # elif q == 'eig':
                #     eig_j = expected_information_gain_multiclass_vectorized(
                #         self.alpha, self.beta,
                #         pred_classes.T, # (N,H)
                #         self.item_priors,
                #         item_idx=j,
                #         num_draws=500
                #     )
                #     _q_values[i] = eig_j


        # _q_values = q_values[_d_u_idxs]
        # q_values = q_values[self.d_u_idxs]
        # print("_q_values", _q_values)
        print("Q (top 10 candidates):", torch.topk(_q_values, k=min(10, len(_q_values))))
        
        if self.stochastic:
            _q_values -= _q_values.min() - 1e-9
            sum_q = _q_values.sum()
            if sum_q < 1e-12:
                print("!!! sum_eig < 1e-12 !!!")
                q_candidates = torch.full(len(_q_values), 1.0 / len(_q_values), device=self.device)
            else:
                q_candidates = _q_values / sum_q
            print("Selection probabilities (top 10):", torch.topk(q_candidates, k=10)[0])
            
            chosen_idx = random.sample(_d_u_idxs, k=1, weights=q_candidates.cpu().numpy().tolist())[0]
            chosen_q = q_candidates[_d_u_idxs.index(chosen_idx)]
        else:
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
        pred_classes = torch.argmax(self.dataset.pred_logits, dim=-1)
        outcome = pred_classes[:, chosen_idx] == true_class
        print(outcome.sum().item(), "models got it right")
        
        w_new = 1.0
        if self.stochastic and self.importance_weighting and selection_prob is not None:
            w_new = (1.0 / len(self.d_u_idxs)) / selection_prob
            print("Importance weight for chosen candidate", (1.0 / len(self.d_u_idxs)), "/", selection_prob.item(), "=", w_new.item())
            _w_new = torch.clip(w_new, min=0.1, max=10.0)
            if _w_new != w_new:
                print("Clipped to", _w_new.item())
                w_new = _w_new
        
        self.alpha, self.beta = beta_update(self.alpha, 
                                            self.beta, 
                                            outcome.to(torch.int), 
                                            weight=self.update_strength*w_new)
        
        self.d_l_idxs.append(chosen_idx)
        self.d_u_idxs.remove(chosen_idx)
        self.qms.append(selection_prob)
        self.d_l_ys.append(true_class)

    def get_best_model_prediction(self):
        posterior_means = self.alpha / (self.alpha + self.beta)
        print("Posterior means shape and sample", posterior_means.shape, posterior_means[0])
        print("Top posterior means", torch.topk(posterior_means, k=min(10, self.H))[0])

        if self.select == 'means':
            best_model_idx_pred = torch.argmax(posterior_means)
        elif self.select == 'sample':
            prob_best = sample_is_best_worker(self.alpha, self.beta, num_draws=1000)
            best_model_idx_pred = torch.argmax(prob_best)
            print(f"After update, best model index (by {self.select}):", best_model_idx_pred, prob_best.max())
            print("Top 10:", torch.topk(prob_best, k=min(10, self.H))[0])
        elif self.select == 'ensemble':
            prob_best = sample_is_best_worker(self.alpha, self.beta, num_draws=1000)
            weights = prob_best
            ensemble = WeightedEnsemble(self.dataset.pred_logits)
            preds = ensemble.get_preds(weights=weights)
            pred_losses = metrics.simple_expected_error(preds, self.dataset.pred_logits).mean(dim=-1)
            best_model_idx_pred = torch.argmin(pred_losses)

            print(f"After update, best model index (by {self.select}):", best_model_idx_pred, prob_best.max(), pred_losses.min())
            print("Top 10 pbest:", torch.topk(prob_best, k=min(10, self.H))[0])

        return best_model_idx_pred
    
    def get_bma_preds(self):
        prob_best = sample_is_best_worker(self.alpha, self.beta, num_draws=1000)
        ensemble = WeightedEnsemble(self.dataset.pred_logits)
        preds = ensemble.get_preds(weights=prob_best)
        return preds

    def get_risk_estimates(self):
        preds = self.get_bma_preds()
        pred_losses = metrics.simple_expected_error(preds, self.dataset.pred_logits).mean(dim=-1)
        return pred_losses
    
    def get_additional_to_log(self):
        return {
            'alpha': self.alpha.cpu(),
            'beta': self.beta.cpu(),
            'posterior entropy': distribution_entropy(sample_is_best_worker(self.alpha, self.beta, num_draws=1000)).cpu()
        }

