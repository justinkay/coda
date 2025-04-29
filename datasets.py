import torch
import os

class Dataset:
    def __init__(self, filepath, device):
        self.device = device
        self.preds = torch.load(filepath, map_location=device)
        print("Loaded preds of shape", self.preds.shape)

        self.labels = None
        label_p = filepath.replace('.pt', '_labels.pt')
        if os.path.exists(label_p):
            self.labels = torch.load(label_p, map_location=device)
            print("Loaded labels of shape", self.labels.shape)
        else:
            print("Did not load labels.")