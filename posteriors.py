import numpy as np
from math import comb
import matplotlib.pyplot as plt
from scipy.stats import beta
import torch


def distribution_entropy(prob: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """
    Compute entropy of a discrete probability distribution 'prob' shape (K,).
    H(p) = - sum_k p_k * log(p_k).
    Returns a 0D tensor (scalar).
    """
    prob_clamped = prob.clamp(min=eps)
    return - (prob_clamped * prob_clamped.log()).sum()


def sample_is_best_worker(alpha: torch.Tensor,
                          beta: torch.Tensor,
                          num_draws=1000,
                          random_state=0) -> torch.Tensor:
    """
    Given alpha[k], beta[k] for K workers (Beta posterior parameters) as PyTorch tensors,
    approximate the probability that each worker is the "best" by Monte Carlo sampling.

    alpha, beta: shape (K,) on the same device (CPU or CUDA).
    Returns prob_best: shape (K,) on the same device.
    """
    torch.manual_seed(random_state)  # Optional for reproducible draws
    dist = torch.distributions.Beta(alpha, beta)  # batch-size = K
    # samples => (num_draws, K)
    samples = dist.sample((num_draws,))

    # For each draw, find which worker is best
    best_idxs = torch.argmax(samples, dim=1)  # shape (num_draws,)

    K = alpha.shape[0]
    best_counts = torch.zeros(K, dtype=torch.int64, device=alpha.device)
    best_counts.index_add_(
        0,
        best_idxs,
        torch.ones_like(best_idxs, dtype=torch.int64)
    )

    prob_best = best_counts.float() / float(num_draws)
    return prob_best

def batch_update_beta_for_item(
    alpha: torch.Tensor,  # shape (K,)
    beta: torch.Tensor,   # shape (K,)
    worker_preds: torch.Tensor,  # shape (K,) => worker_preds for item i
    C: int
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

    # For each worker k, find which class is predicted => worker_preds[k]
    # We can do a vector eq check: eq_classes[c, k] = (worker_preds[k] == c)
    # shape => (C, K)
    pred_classes = worker_preds.unsqueeze(0).expand(C, K_)  # replicate rowwise
    class_range  = torch.arange(C, device=alpha.device).unsqueeze(1).expand(C, K_)
    eq_mask = (pred_classes == class_range)  # shape (C, K) => True if worker_preds[k] == c

    # eq_mask => True => alpha++ for that worker c
    # eq_mask => False => beta++ for that worker c
    # We'll do alpha_batch[eq_mask] += 1, alpha_batch => shape (C,K)
    alpha_batch[eq_mask] += 1.0
    beta_batch[~eq_mask]  += 1.0

    return alpha_batch, beta_batch


def sample_is_best_worker_beta_batched(
    alpha_batch: torch.Tensor, # shape (C,K)
    beta_batch: torch.Tensor,  # shape (C,K)
    num_draws=1000,
    random_state=0
):
    """
    Single call to sample Beta distributions for each row (C*K).
    Then compute who is best in each scenario c.

    Returns:
      prob_best_c: shape (C, K), i.e. prob_best_c[c] is distribution for class c
    """
    device = alpha_batch.device
    torch.manual_seed(random_state)

    C_, K_ = alpha_batch.shape
    # Flatten => (C*K,)
    alpha_1d = alpha_batch.view(C_*K_)
    beta_1d  = beta_batch.view(C_*K_)

    # Build a batch Beta distribution over (C*K) parameters
    dist = torch.distributions.Beta(alpha_1d, beta_1d)
    # samples => shape (num_draws, C*K)
    samples_2d = dist.sample((num_draws,))

    # Reshape => (num_draws, C, K)
    samples_3d = samples_2d.view(num_draws, C_, K_)

    # For each draw, each c, find which worker is best
    # shape => (num_draws, C)
    best_idxs = torch.argmax(samples_3d, dim=-1)

    # Tally best counts => shape (C, K)
    best_counts = torch.zeros(C_, K_, dtype=torch.int64, device=device)

    for c in range(C_):
        c_best = best_idxs[:, c]      # shape (num_draws,)
        c_counts = torch.zeros(K_, dtype=torch.int64, device=device)
        c_counts.index_add_(
            0,
            c_best,
            torch.ones_like(c_best, dtype=torch.int64)
        )
        best_counts[c] = c_counts

    # Probability each worker is best => shape (C, K)
    prob_best_c = best_counts.float() / float(num_draws)

    return prob_best_c

def expected_worker_differentiation_multiclass_vectorized(
    alpha: torch.Tensor,        # shape (K,)
    beta: torch.Tensor,         # shape (K,)
    worker_preds: torch.Tensor, # shape (N, K) => integer class predictions
    p_item: torch.Tensor,       # shape (N, C)
    item_idx: int,
    num_draws=1000,
    random_state=0
) -> float:
    """
    Vectorized multi-class Beta-Bernoulli approach:
      - current_prob_best = sample_is_best_worker(alpha, beta)
      - alpha_batch,beta_batch = batch_update_beta_for_item(...) => shape (C,K)
      - prob_best_c = sample_is_best_worker_beta_batched(alpha_batch,beta_batch)
      - for each c, measure L1(prob_best_c[c] vs current_prob_best), weigh by p_item[item_idx,c]

    Returns a float for the expected difference in "best worker" distribution.
    """

    # 1) Current distribution
    current_prob_best = sample_is_best_worker(alpha, beta, num_draws=num_draws, random_state=random_state)
    
    # 2) Build alpha_batch,beta_batch => shape (C,K)
    N, C_ = p_item.shape
    K_ = alpha.shape[0]
    p_vec = p_item[item_idx]  # shape (C,)
    
    alpha_batch, beta_batch = batch_update_beta_for_item(
        alpha, beta,
        worker_preds[item_idx],  # shape (K,)
        C=C_
    )

    # 3) single pass => (C, K)
    prob_best_c = sample_is_best_worker_beta_batched(
        alpha_batch, beta_batch, 
        num_draws=num_draws,
        random_state=random_state+999
    )

    # 4) Weighted L1 difference
    dist_sum = 0.0
    for c in range(C_):
        pc_val = p_vec[c].item()
        if pc_val < 1e-12:
            continue
        
        diff_c = torch.sum(torch.abs(prob_best_c[c] - current_prob_best)).item()
        # JK semi-hack: weight by current_prob_best to prioritize models that were ranked highly last iter
        # this could be counterproductive if current_prob_best is bad
        # changes = torch.abs(prob_best_c[c] - current_prob_best)
        # diff_c = torch.dot(current_prob_best, changes)

        dist_sum += pc_val * diff_c

    return dist_sum

def expected_information_gain_multiclass_vectorized(
    alpha: torch.Tensor,      # shape (K,)
    beta:  torch.Tensor,      # shape (K,)
    worker_preds: torch.Tensor, # shape (N, K)
    p_item: torch.Tensor,     # shape (N, C) => p_item[i,c]
    item_idx: int,
    num_draws=1000,
    random_state=0
) -> float:
    """
    Vectorized multi-class Beta-Bernoulli approach, measuring
    *expected reduction in entropy* of the "best worker" distribution.

    Steps:
      1) current_prob_best = sample_is_best_worker(alpha, beta)
      2) H_current = distribution_entropy(current_prob_best)
      3) batch_update_beta_for_item => alpha_batch, beta_batch => (C,K)
      4) sample_is_best_worker_beta_batched => prob_best_c => (C,K)
      5) for each class c => H_c = distribution_entropy(prob_best_c[c]),
         info_gain_c = H_current - H_c
         weigh by p_item[i,c]
      6) sum => total info gain

    Returns a Python float.
    """
    # 1) current distribution
    from math import isclose

    torch.manual_seed(random_state)
    current_prob_best = sample_is_best_worker(alpha, beta, num_draws=num_draws, random_state=random_state)
    
    # 2) measure current entropy
    H_current = distribution_entropy(current_prob_best)

    # 3) build alpha/beta batch
    _, C_ = p_item.shape
    p_vec = p_item[item_idx]  # shape (C,)
    alpha_batch, beta_batch = batch_update_beta_for_item(
        alpha, beta,
        worker_preds[item_idx],  # shape (K,)
        C_
    )

    # 4) sample => prob_best_c => shape (C,K)
    prob_best_c = sample_is_best_worker_beta_batched(
        alpha_batch, beta_batch,
        num_draws=num_draws,
        random_state=random_state+999
    )

    # 5) compute expected reduction in entropy
    dist_sum = 0.0
    for c in range(C_):
        pc_val = p_vec[c].item()
        if pc_val < 1e-12:
            continue
        H_c = distribution_entropy(prob_best_c[c])
        info_gain_c = (H_current - H_c).item()
        dist_sum += pc_val * info_gain_c

    return dist_sum


def select_most_informative_item_multiclass(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    candidate_indices,
    num_draws=1000
):
    """
    Among candidate_indices, pick the item with max 'expected_worker_differentiation_multiclass'.
    """
    from tqdm import tqdm
    best_item = None
    best_value = -1.0
    for i in tqdm(candidate_indices, desc='Computing most informative item (multiclass)'):
        val = expected_worker_differentiation_multiclass_vectorized(alpha, beta, worker_preds, p_item, i, num_draws=num_draws)
        if val > best_value:
            best_value = val
            best_item = i
    return best_item, best_value


def select_most_informative_item_EIG_multiclass(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    worker_preds: torch.Tensor,
    p_item: torch.Tensor,
    candidate_indices,
    num_draws=1000
):
    from tqdm import tqdm
    best_item = None
    best_value = -1.0
    for i in tqdm(candidate_indices, desc='Computing most informative item (EIG multiclass)'):
        value = expected_information_gain_multiclass_vectorized(
            alpha, beta,
            worker_preds,
            p_item,
            item_idx=i,
            num_draws=num_draws
        )
        if value > best_value:
            best_value = value
            best_item = i
    return best_item, best_value


# a fast heuristic for filtering down to n data points based on disagreement between the top k workers
def weighted_disagreement_candidates(preds, prob_best, k=5, n=50):
    """
    Preds: (N, H); dataset_tensor.argmax(dim=-1) 
    """
    # 1) pick top k workers by prob_best
    top_probs, topk_idxs = torch.topk(prob_best, k=k)
    # shape topk_idxs => (k,), top_probs => (k,)

    # 2) get worker predictions for these top k
    #    dataset_tensor shape => (N, K, C)
    #    we can do argmax across C for each worker
    # preds = dataset_tensor.argmax(dim=-1)  # shape (N, K)
    # subset => shape (N, k)
    preds_top = preds[:, topk_idxs]

    # 3) for each item i, measure how many top workers disagree
    # weigh each worker by top_probs
    # e.g. "score(i) = sum_{w} prob_best[w] * 1 if pred differs from majority"

    N = preds_top.shape[0]
    scores = torch.zeros(N, device=preds.device)

    for i in range(N):
        # predictions from top workers => shape (k,)
        item_preds = preds_top[i]
        # simplest measure: pairwise or majority-based
        # let's do a "proportion that is not the majority class"
        vals, counts = torch.unique(item_preds, return_counts=True)
        majority_count = counts.max()
        majority_class = vals[counts.argmax()]
        # sum prob_best[w] for each worker w who is not majority
        not_majority_mask = (item_preds != majority_class)
        # gather their probabilities
        idxs = torch.nonzero(not_majority_mask).flatten()
        scores[i] = top_probs[idxs].sum()

    # 4) pick top-n
    _, topn_indices = torch.topk(scores, n)
    return topn_indices

############ These are slow:

# ----------------------- Beta–Binomial Updates -----------------------
def beta_update(a, b, outcome, weight=1.0):
    """
    Conjugate Beta update for a single item (n = 1 per model).
    outcome: binary vector of shape (M,)
    weight: importance weight for this item.
    Returns updated (a, b) arrays.
    """
    return a + weight * outcome, b + weight * (1 - outcome)

# ----------------------- Global Bias Adjustment -----------------------
def global_bias_adjustment(a, b, target_rate):
    """
    Empirical Bayes adjustment: adjust each model’s Beta parameters so that
    the average posterior mean moves toward the target_rate.
    
    Let mu_i = a_i/(a_i+b_i), and overall_mean = mean(mu_i).
    Then update:
       a_i_adj = a_i * (target_rate / overall_mean)
       b_i_adj = b_i * ((1 - target_rate) / (1 - overall_mean))
    """
    mu = a / (a + b)
    overall_mean = np.mean(mu)
    # Avoid division by zero:
    if overall_mean < 1e-12 or overall_mean > 1 - 1e-12:
        return a, b
    a_adj = a * (target_rate / overall_mean)
    b_adj = b * ((1 - target_rate) / (1 - overall_mean))
    return a_adj, b_adj

# ----------------------- Sampling from Beta Distributions -----------------------
def sample_beta_best(a, b, num_draws=1000, rng=None):
    """
    Given Beta parameters a and b for each model (arrays of shape (M,)),
    sample num_draws samples from each Beta, and compute the probability each model is best.
    Returns an array of shape (M,) with estimated probabilities.
    """
    M = a.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    # Draw samples for each model; result shape (num_draws, M)
    samples = rng.beta(a, b, size=(num_draws, M))
    best_idx = np.argmax(samples, axis=1)
    counts = np.bincount(best_idx, minlength=M)
    return counts / num_draws

def discrete_entropy(prob, eps=1e-12):
    prob = np.asarray(prob)
    prob = np.clip(prob, eps, 1)
    return -np.sum(prob * np.log2(prob))

# ----------------------- EIG Computation -----------------------
def compute_eig_for_candidate_bb(a, b, candidate_index, candidate_predictions, candidate_prior,
                                 weight_for_eig=1.0, num_draws=1000, rng=None):
    """
    For candidate item j (identified by candidate_index in {0,...,K-1}),
    given the current Beta parameters a, b (each of shape (M,)),
    and a lookup matrix candidate_predictions (shape: M x K) that gives each model’s predicted class on candidate j,
    and candidate_prior: an array of shape (C,) representing the prior over true classes for this item,
    compute the expected reduction in entropy (EIG) if we were to update with that candidate.
    
    For each possible true class c in {0, ..., C-1}:
      - Outcome vector: outcome[i] = 1 if candidate_predictions[i, candidate_index] == c else 0.
      - Updated Beta parameters: a_new = a + weight_for_eig * outcome,  b_new = b + weight_for_eig * (1 - outcome)
      - Compute new best-model distribution from the updated parameters via Monte Carlo sampling.
      - Compute the entropy H(c) of that distribution.
    
    Let current_entropy = H_old computed from current a, b.
    Expected new entropy = sum_c candidate_prior[c] * H(c).
    EIG = H_old - Expected new entropy.
    
    Returns a scalar EIG.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    M = a.shape[0]
    C = candidate_prior.shape[0]
    
    # Compute current best-model distribution entropy
    current_prob = sample_beta_best(a, b, num_draws=num_draws, rng=rng)
    H_old = discrete_entropy(current_prob)
    
    expected_H = 0.0
    # For each possible true class c:
    for c in range(C):
        p_c = candidate_prior[c]
        if p_c < 1e-12:
            continue
        # Outcome: outcome[i] = 1 if candidate_predictions[i, candidate_index] == c else 0.
        outcome = np.array([1 if candidate_predictions[i, candidate_index] == c else 0 for i in range(M)])
        # Updated parameters (using a notional weight for EIG evaluation)
        a_new = a + weight_for_eig * outcome
        b_new = b + weight_for_eig * (1 - outcome)
        new_prob = sample_beta_best(a_new, b_new, num_draws=num_draws, rng=rng)
        H_new = discrete_entropy(new_prob)
        expected_H += p_c * H_new
    return H_old - expected_H

def eig_q(a, # (H,), beta distribution alphas
          b, # (H,), beta distribution betas
          candidate_predictions, # (M, H) 
          candidate_prior,       # (M, C) class priors
          rng=None):
    M, H = candidate_predictions.shape
    eig_values = np.zeros(H)
    for j in range(H):
        eig_j = compute_eig_for_candidate_bb(a, b, j, candidate_predictions,
                                                candidate_prior, weight_for_eig=1.0,
                                                num_draws=500, rng=rng)
        eig_values[j] = eig_j
    # Display some summary stats
    print("EIG (first 10 candidates):", eig_values[:10])
    
    # account for negative values
    eig_values -= eig_values.min() - 1e-3

    # Randomize among candidates: define selection probability proportional to EIG.
    # TODO softmax?
    sum_eig = np.sum(eig_values)
    if sum_eig < 1e-12:
        q_candidates = np.full(H, 1.0 / H)
    else:
        q_candidates = eig_values / sum_eig
    print("Selection probabilities (first 10):", q_candidates[:10])

    return q_candidates