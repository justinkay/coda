import numpy as np
from io import BytesIO
from typing import Sequence, Union
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:  # pillow might not be installed at runtime
    Image = None


def plot_bar(values: Union[Sequence[float], 'np.ndarray', 'torch.Tensor']):
    """Return a PIL Image visualizing a simple bar chart."""
    try:
        import torch
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
    except Exception:
        pass

    values = np.asarray(values)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(range(len(values)), values)
    ax.set_xticks([])

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    if Image is not None:
        return Image.open(buf)
    return buf.getvalue()
