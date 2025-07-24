import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Ensemble:
    def __init__(self, preds, **kwargs):
        self.preds = preds
        self.device = preds.device
        H, N, C = preds.shape

    def get_preds(self, **kwargs):
        return self.preds.mean(dim=0)
    

def distribution_entropy(prob: torch.Tensor, eps=1e-12) -> torch.Tensor:
    prob_clamped = prob.clamp(min=eps)
    return -(prob_clamped * prob_clamped.log2()).sum()


def _check(t: torch.Tensor, name: str, *, raise_err=True):
    # for debugging - check t for infs/nans
    bad = ~torch.isfinite(t)
    if bad.any():
        msg = (f"[NUMERIC ERROR] {name} has {bad.sum()} bad values "
            f"(NaN/Inf) out of {t.numel()} "
            f"min={t.min().item():.3g}, max={t.max().item():.3g}")
        if raise_err:
            raise RuntimeError(msg)
        print(msg)


def _check_prob(p: torch.Tensor, name="prob", eps=1e-12):
    # check if p is a valid probability distribution
    _check(p, name)
    if (p < -eps).any():
        raise RuntimeError(f"{name} has negatives")
    s = p.sum(-1)
    if (torch.isnan(s) | torch.isinf(s)).any():
        raise RuntimeError(f"{name} sum is nan/inf")
    if ((s - 1).abs() > 1e-4).any():
        print(f"[WARN] {name} rows not normalised: min sum={s.min():.4f}, "
            f"max sum={s.max():.4f}")


def plot_bar(data, fig_size=(10, 5), title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=fig_size)

    # convert to numpy array on cpu if not already
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array, list, or torch tensor.")

    data = data.squeeze()
    print('data.shape[0]', data.shape[0])
    ax.bar(list(range(data.shape[0])), data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

    plt.close(fig)

    return img