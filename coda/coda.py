import torch
from torch.distributions import Beta
import torch.nn.functional as F
from tqdm import tqdm
import random

from coda.base import ModelSelector
from surrogates import Ensemble


def distribution_entropy(prob: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """
    Compute entropy of a discrete probability distribution 'prob' shape (K,).
    H(p) = - sum_k p_k * log(p_k).
    Returns a 0D tensor (scalar).
    """
    prob_clamped = prob.clamp(min=eps)
    return - (prob_clamped * prob_clamped.log2()).sum()

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

def compute_p_best_dir_beta_mom(
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
    return compute_p_best_betas(alpha_prime, beta_prime, num_points)

def compute_p_best_betas(alpha: torch.Tensor, beta: torch.Tensor, num_points=128) -> torch.Tensor:
    device = alpha.device

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

def compute_p_best_dirichlet(
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
    return compute_p_best_dir_beta_mom(
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

def batch_update_dirichlet_for_item(
    dirichlet_alphas: torch.Tensor,  # (H, C, C)
    worker_preds: torch.Tensor,      # (N, H, C) - soft predictions
    update_weight: float = 1.0
) -> torch.Tensor:
    """
    Soft update of Dirichlet parameters using probabilistic worker predictions
    """
    N, H, C = worker_preds.shape
    
    # 1. Create N*C hypothetical confusion matrices - one per item and possible class
    #    (H, C, C) -> (N, C, H, C, C)
    updated = dirichlet_alphas.unsqueeze(0).unsqueeze(1).expand(N, C, H, C, C).clone()

    # 2. Duplicate (one-hot) worker predictions C times to make broadcasting easier
    #    (N, H, C) -> (N, C, H, C)
    updates = worker_preds.unsqueeze(1).expand(-1, C, -1, -1) * update_weight

    # 3. Do the hypothetical updates
    for c in range(C):
        updated[:, c, :, c, :] += updates[:, c, :, :]

    return updated

def compute_p_best_beta_batched(
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

def eig_dirichlet_batched(
    dirichlet_alphas: torch.Tensor,
    worker_preds: torch.Tensor,
    marginal_distribution: torch.Tensor,
    candidate_ids: list[int],
    chunk_size: int = 100,
    update_weight = 1.0,
    update_rule="hard"
) -> torch.Tensor:
    device = dirichlet_alphas.device
    _, H, C = worker_preds.shape
    candidates = torch.tensor(candidate_ids, device=device)
    
    # get current P_best distribution and entropy
    current_probs = compute_p_best_dirichlet(dirichlet_alphas, marginal_distribution)
    H_current = distribution_entropy(current_probs)

    # sanity checks
    print("model 0 dirichlet 0 sum", dirichlet_alphas[0][0].sum())
    print("marginal sum", marginal_distribution.sum())
    print("current probs sum", current_probs.sum())
    print("H current", H_current)
    
    eig_results = []
    # batch over N (items / data points)
    for chunk_start in tqdm(range(0, len(candidates), chunk_size)):
        chunk = candidates[chunk_start:chunk_start+chunk_size] # get idxs of this batch
        chunk_probs = worker_preds[chunk]
        
        if update_rule == 'hard':
            chunk_probs = F.one_hot(torch.argmax(chunk_probs, dim=-1), num_classes=C).float()

        # Updated dirichlets shape: (chunk_size, C, H, C, C)
        updated_dirichlets = batch_update_dirichlet_for_item(
            dirichlet_alphas, 
            chunk_probs, 
            update_weight=update_weight
        )
        if chunk_start == 0:
            print("model 0 dirichlet 0 sum after batch update", updated_dirichlets[0][0][0][0].sum())

        # Flatten the candidate and model dimensions together:
        flat_dirichlets = updated_dirichlets.reshape(chunk_size * C * H, C, C)  # shape: (B * C * H, C, C)

        # Convert each confusion matrix (of shape (C, C)) to a Beta distribution
        alpha_beta_flat, beta_beta_flat = dirichlet_to_beta_with_marginal(flat_dirichlets, marginal_distribution)
        # These have shape (chunk_size * C * H,)

        # Now reshape the output back to (chunk_size, C, H)
        alpha_beta_batch = alpha_beta_flat.reshape(chunk_size, C, H)
        beta_beta_batch = beta_beta_flat.reshape(chunk_size, C, H)
        
        if chunk_start == 0:
            print("model 0 class 0 alpha beta after reshape", alpha_beta_batch[0][0][0], beta_beta_batch[0][0][0])
            print("alpha_beta_batch.shape", alpha_beta_batch.shape, beta_beta_batch.shape)
            print("nan check", torch.any(torch.isnan(alpha_beta_batch)), torch.any(torch.isnan(beta_beta_batch)))
            print("negative check", torch.any(alpha_beta_batch < 0), torch.any(beta_beta_batch < 0))

        # Rest of the workflow remains the same
        updated_probs = compute_p_best_beta_batched(
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

class CODA(ModelSelector):
    def __init__(self, dataset,
                prior_source="ens",
                q='eig',                       # acquisition function
                prefilter_fn='disagreement',   # filter with heuristic
                prefilter_n=0,                 # number to filter down to
                epsilon=0.0,                   
                update_rule="hard",
                base_prior="diag",

                # new
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
        self.update_rule = update_rule
        self.base_prior = base_prior

        # NEW
        self.base_strength = alpha / temperature
        self.prior_strength = (1 - alpha) / temperature
        self.update_strength = learning_rate_ratio / temperature
        self.hypothetical_update_strength = 1.0 # fix this for sanity

        # initialize dirichlets (confusion matrices) and class marginals
        ensemble_preds = torch.argmax(Ensemble(dataset.preds).get_preds(), dim=-1)
        pred_classes = torch.argmax(dataset.preds, dim=-1)  # (H, N)
        if prior_source == "ens" or prior_source == "ens-exp":
            preds_soft = dataset.preds # F.softmax(dataset.pred_logits, dim=-1)
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
        self.d_u_idxs = list(range(dataset.preds.shape[1]))
        self.qms = []
        self.stochastic = False

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
                    temperature=args.temperature,
                    alpha=args.alpha,
                    learning_rate_ratio=args.learning_rate_ratio,
                    base_prior=args.base_prior
                )

    def get_next_item_to_label(self):
        pred_classes = torch.argmax(self.dataset.preds, dim=-1)

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
                self.stochastic = True
            else:
                raise NotImplemented

        if len(_d_u_idxs) > self.prefilter_n and self.prefilter_n > 0:
            _d_u_idxs = random.sample(_d_u_idxs, k=self.prefilter_n)
            self.stochastic = True

        q = self.q

        if len(_d_u_idxs) == 0:
            print("Prefiltered all points; falling back to random sampling")
            q = 'iid'

        if self.epsilon and random.random() < self.epsilon:
            print("Epsilon sample")
            q = 'iid'

        if q == 'iid':
            _q_values = torch.ones(len(_d_u_idxs), device=self.device) * 1 / len(_d_u_idxs)
            self.stochastic = True
        elif q =='eig':
            dataset_tensor = self.dataset.preds.permute(1,0,2) #F.softmax(self.dataset.pred_logits.permute(1,0,2), dim=-1)
            _q_values = eig_dirichlet_batched(
                                self.dirichlets, 
                                dataset_tensor,
                                self.curr_marginal,
                                _d_u_idxs,
                                chunk_size=4,
                                update_rule=self.update_rule,
                                update_weight=self.hypothetical_update_strength
            )
        elif q == 'uncertainty':
            pred_probs = self.dataset.preds # F.softmax(self.dataset.pred_logits, dim=2)
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
            self.stochastic = True
        # chosen_idx = self.d_u_idxs[chosen_local_idx.item()]
        chosen_idx = _d_u_idxs[chosen_local_idx.item()]
        chosen_q = chosen_q_val

        print("Chosen q, candidate index:", chosen_q, chosen_idx)
        return chosen_idx, chosen_q
    
    def add_label(self, chosen_idx, true_class, selection_prob):
        true_class = int(true_class) # just in case; e.g. GLUE preds break this
        preds_soft = self.dataset.preds # F.softmax(self.dataset.pred_logits, dim=-1)
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
        prob_best = compute_p_best_dirichlet(self.dirichlets, self.curr_marginal)

        if torch.isnan(prob_best).any():
            raise ValueError("NaN in posterior")

        best_model_idx_pred = torch.argmax(prob_best)
        print(f"After update, best model index:", best_model_idx_pred, prob_best.max())
        print("Top 10:", torch.topk(prob_best, k=min(10, self.H))[0]) 

        return best_model_idx_pred