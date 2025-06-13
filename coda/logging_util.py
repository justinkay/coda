import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np


def plot_bar(data, fig_size=(10, 5), title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=fig_size)

    # convert to numpy array on cpu if not already
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array, list, or torch tensor.")

    ax.bar(range(len(data)), data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

    plt.close(fig)

    return img