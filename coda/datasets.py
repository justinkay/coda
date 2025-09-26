import torch
import os

class Dataset:
    """
    A model selection dataset is a tensor of shape (H,N,C) containing post-softmax prediction scores, 
    where H is the number of models, N is the number of datapoints, and C is the number of classes.

    Optionally, it can also contain an (N,) shaped matrix (assumed to be a file appended with '_labels.pt')
    of ground-truth class labels.
    """
    def __init__(self, filepath, device):
        self.device = device
        self.preds = torch.load(filepath, map_location=device).float() # avoid fp16 precision errors
        print("Loaded preds of shape", self.preds.shape)

        self.labels = None
        label_p = filepath.replace('.pt', '_labels.pt')
        if os.path.exists(label_p):
            self.labels = torch.load(label_p, map_location=device)
            print("Loaded labels of shape", self.labels.shape)
        else:
            print("Did not load labels.")