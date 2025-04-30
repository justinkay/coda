import torch
import torch.nn.functional as F


def simple_expected_error(surrogate_preds, preds):
    """
    Args:
        surrogate_preds: (N, C) post-softmax
        pred_logits: (H,N,C)

    Get expected error (``accuracy losses'') of some other predictions (pred_logits) compared to this surrogate.

    Return shape: (H,N)
    """
    H, N, C = preds.shape

    pred_probs = preds # torch.softmax(pred_logits, dim=-1)
    pred_classes = pred_probs.argmax(dim=-1)

    y_star_probs = surrogate_preds[torch.arange(surrogate_preds.shape[0]), pred_classes]
    exp_accuracy_loss = 1 - y_star_probs

    return exp_accuracy_loss

def simple_error(surrogate_preds, preds):
    """
    Args:
        surrogate_preds: (N, C) post-softmax
        pred_logits: (H,N,C)

    Get error of some other predictions (pred_logits) compared to this surrogate based on 
    argmax class predictions of surrogate.

    Return shape: (H,)
    """
    H, N, C = preds.shape
    ensemble_pred_classes = surrogate_preds.argmax(dim=-1)
    model_pred_classes = preds.argmax(dim=-1)
    accs = (model_pred_classes == ensemble_pred_classes).float().mean(dim=-1)
    return 1 - accs

def ece_multi(probs, correct, nbin=30, fn=torch.abs):
    """
    Calculate ECE for multiple models in PyTorch.
    :param probs: (torch.Tensor) probability predictions (max probability), shape (H, N)
    :param correct: (torch.Tensor) indicator whether prediction was true, shape (H, N)
    :param nbin: (int) number of bins for calculating ECE
    :param fn: (function) function to transform conf - acc to fn(conf - acc) for ECE
    :return: (torch.Tensor, torch.Tensor, torch.Tensor) ece_correct, ece_total, deviation
    """
    H, N = probs.shape
    bins = (probs * nbin).long()
    bins = torch.clamp(bins, max=nbin - 1)  # Ensure bins are within [0, nbin - 1]

    # Create a one-hot encoding of the bins
    bin_mask = F.one_hot(bins, num_classes=nbin)  # Shape: (H, N, nbin)

    # Calculate total counts per bin per model
    total_per_bin = bin_mask.sum(dim=1)  # Shape: (H, nbin)

    # Expand correct tensor to match bin_mask dimensions
    correct_expanded = correct.unsqueeze(2)  # Shape: (H, N, 1)

    # Calculate correct counts per bin per model
    correct_per_bin = (bin_mask * correct_expanded).sum(dim=1)  # Shape: (H, nbin)

    # Compute accuracy per bin per model
    acc = torch.where(
        total_per_bin > 0, 
        correct_per_bin.float() / total_per_bin.float(), 
        torch.tensor(-1.0).to(probs.device)
    )

    # Expand probs tensor to match bin_mask dimensions
    probs_expanded = probs.unsqueeze(2)  # Shape: (H, N, 1)

    # Compute confidence per bin per model
    conf_numerators = (bin_mask * probs_expanded).sum(dim=1)  # Shape: (H, nbin)
    conf = torch.where(
        total_per_bin > 0, 
        conf_numerators / total_per_bin.float(), 
        torch.tensor(0.0).to(probs.device)
    )

    # Calculate deviation per bin per model
    deviation_numerators = torch.where(
        acc >= 0,
        fn(acc - conf) * total_per_bin.float(),
        torch.tensor(0.0).to(probs.device)
    )  # Shape: (H, nbin)

    # Sum deviations across bins for each model
    deviation_h = deviation_numerators.sum(dim=1)  # Shape: (H,)

    # Total counts per model
    deviation_denominator = total_per_bin.sum(dim=1)  # Shape: (H,)

    # Final ECE deviation per model
    ece = deviation_h / deviation_denominator

    return ece, correct_per_bin, total_per_bin

def ece(predictions, correct, nbin=30, fn=torch.abs):
    """
    Args:
        predictions: (H,N,C) post-softmax
        correct: ground truth labels (N,)
    
    Return:
        ece: (H,)

    Don't return (for now):
        ece_correct: (H, nbin)
        ece_total: (H, nbin)
    """
    probs, pred_classes = torch.max(predictions, dim=2)  # probs: (H, N), pred_classes: (H, N)

    correct_labels = correct.unsqueeze(0).expand(predictions.shape[0], -1)  # Shape: (H, N)
    correct_tensor = (pred_classes == correct_labels).float()  # Shape: (H, N)
    ece, correct_per_bin, total_per_bin = ece_multi(probs, correct_tensor, nbin=nbin, fn=fn)

    return ece