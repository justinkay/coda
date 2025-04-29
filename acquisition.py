import torch
import torch.nn.functional as F


def lure_acquisition_ce(pred_logits, ensemble_pred_logits):
    """
    Cross entropy acquisition function from Active Testing eq. 12.

    q(im) ∝ - sum_y π(y | x_im) log f(x_im)_y,

    Args:
    - pred_logits: Tensor of shape (H0, N, C) representing raw logits
                        from each model for each data point.
    - ensemble_pred_logits: Tensor of shape (H1, N, C) representing raw logits
                        from each model in the ensemble for each data point.

    Note: returns separate acquisition fn for each model in H0.
    """
    # Apply softmax to ensemble predictions to convert them to probabilities
    ensemble_probs = F.softmax(ensemble_pred_logits, dim=2)  # Shape (H1, N, C)

    # Compute the marginal π(y | x_n) by averaging over the ensemble members (mean over H1)
    pi_y = ensemble_probs.mean(dim=0)  # Shape (N, C)

    # Apply log softmax to main model predictions to get log probabilities
    log_f = F.log_softmax(pred_logits, dim=2)  # Shape (H0, N, C)

    # Compute the acquisition function using tensor operations
    # q(h1, n) = - sum_y pi_y * log_f for each main model h0 and data point n
    # Broadcast pi_y (N, C) to match log_f (H0, N, C)
    pi_y_broadcasted = pi_y.unsqueeze(0)  # Shape (1, N, C), broadcastable to (H0, N, C)
    qs = -(pi_y_broadcasted * log_f).sum(dim=2)  # Shape (H0, N)

    return qs

def ams_acquisition_ce_regret(pred_logits, ensemble_pred_logits):
    """
    Compute the expected regret for each data point.

    Args:
    - pred_logits: Tensor of shape (H0, N, C) representing raw logits from each model for each data point.
    - ensemble_pred_logits: Tensor of shape (H1, N, C) representing raw logits from each ensemble model for each data point.

    """
    model_losses = lure_acquisition_ce(pred_logits, ensemble_pred_logits)  # Shape (H0, N)

    # Compute the marginal π(y | x_n) by averaging over the ensemble members (mean over H1)
    ensemble_probs = F.softmax(ensemble_pred_logits, dim=2)
    pi_y = ensemble_probs.mean(dim=0)  # Shape (N, C)

    # Compute the expected loss for f* (best possible model, approximated by the ensemble itself)
    log_pi_y = pi_y.log()  # Shape (N, C)
    best_model_loss = -(pi_y * log_pi_y).sum(dim=1)  # Shape (N,)

    # Step 5: Compute the expected regret
    # expected_regret = model_losses.mean(dim=0) - best_model_loss  # Shape (N,)
    expected_regret = model_losses - best_model_loss  # Shape (N,)

    return expected_regret

def lure_acquisition_acc(pred_logits, ensemble_pred_logits):
    # Compute the surrogate's probabilities
    ensemble_probs = F.softmax(ensemble_pred_logits, dim=2)
    pi_y = ensemble_probs.mean(dim=0)
    
    # Compute the main models' predicted classes
    pred_probs = F.softmax(pred_logits, dim=2)
    pred_classes = pred_probs.argmax(dim=2)
    
    # Get the surrogate's probability for the predicted classes
    y_star_probs = pi_y[torch.arange(pi_y.shape[0]), pred_classes]
    
    # Compute the acquisition score
    acquisition_scores = 1 - y_star_probs
    
    return acquisition_scores

def ams_acquisition_acc_regret(pred_logits, ensemble_pred_logits):
    """
    Accuracy acquisition function that utilizes expected regret, keeping the output shape as (H0, N).

    Args:
    - pred_logits: Tensor of shape (H0, N, C) representing raw logits from each main model for each data point.
    - ensemble_pred_logits: Tensor of shape (H1, N, C) representing raw logits from each ensemble (surrogate) model for each data point.

    Returns:
    - acquisition_scores: Tensor of shape (H0, N) representing the acquisition scores (expected regrets) for each model and data point.
    """
    # Compute the ensemble mean probabilities (surrogate probabilities)
    ensemble_probs = F.softmax(ensemble_pred_logits, dim=2)  # Shape (H1, N, C)
    pi_y = ensemble_probs.mean(dim=0)  # Shape (N, C)

    # Compute the predicted probabilities for the main models
    pred_probs = F.softmax(pred_logits, dim=2)  # Shape (H0, N, C)

    # Get the predicted classes (y*) for each main model and data point
    pred_classes = pred_probs.argmax(dim=2)  # Shape (H0, N)

    # Get the surrogate probabilities for the predicted classes
    H0, N = pred_classes.shape
    y_indices = torch.arange(N, device=pred_classes.device)
    y_star_probs = pi_y[y_indices.unsqueeze(0).expand(H0, N), pred_classes]  # Shape (H0, N)

    # Compute the acquisition scores (losses) for each model and data point
    model_losses = 1 - y_star_probs  # Shape (H0, N)

    # Compute the best possible loss (lowest expected loss) for each data point
    best_model_loss = 1 - pi_y.max(dim=1).values  # Shape (N,)

    # Compute the expected regret for each model and data point
    expected_regret = model_losses - best_model_loss.unsqueeze(0)  # Shape (H0, N)

    # Ensure the expected regret is non-negative
    # expected_regret = torch.clamp(expected_regret, min=0)  # Shape (H0, N)

    return expected_regret

def iid_acquisition(pred_logits, *args):
    H, N, _ = pred_logits.shape
    acquisition_scores = torch.full((H, N), 1 / N, device=pred_logits.device)
    return acquisition_scores