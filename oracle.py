from datasets import DomainNet126
import numpy as np
import torch
import json
import os
import pandas as pd

class Oracle:
    def __init__(self, dataset, base_dir="../powerful-benchmarker/datasets/",
                 loss_fn=None, accuracy_fn=None, **kwargs):
        if isinstance(dataset, DomainNet126):
            test_set = dataset.task.split("_")[-1]
            if dataset.use_target_val:
                test_set_txt = f'{base_dir}/domainnet/{test_set}126_test.txt'
            else:
                test_set_txt = f'{base_dir}/domainnet/{test_set}126_train.txt'
        else:
            raise NotImplementedError()
        
        with open(test_set_txt, 'r') as f:
            self.labels = np.array([ int(s.split(" ")[-1].replace("\n","")) for s in f.readlines()])
            self.labels = torch.tensor(self.labels, device=dataset.device)

        self.dataset = dataset
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.device = dataset.device

    def true_losses(self, pred_logits):
        """
        Compute the mean loss for each model.
        
        Args:
        - preds: Tensor of shape (H, N, C) representing raw logits from each model for each data point.
        
        Returns:
        - Tensor of shape (H,) representing the mean loss for each model.
        """
        H, N, C = pred_logits.shape
        return self.loss_fn(pred_logits.view(-1, C), self.labels.repeat(H), 
                            reduction='none').view(H, N).mean(dim=1)

    def true_accuracies(self, pred_logits):
        H, N, C = pred_logits.shape
        accuracies = []
        batch_size = 100  # Adjust this value based on your GPU memory
        for i in range(0, H, batch_size):
            batch_end = min(i + batch_size, H)
            batch_preds = torch.softmax(pred_logits[i:batch_end], dim=-1)
            batch_labels = self.labels
            batch_acc = []
            for j in range(batch_end - i):
                acc = self.accuracy_fn(batch_preds[j], batch_labels)
                batch_acc.append(acc)
            accuracies.extend([acc.item() for acc in batch_acc])
            torch.cuda.empty_cache()
        return accuracies

    def __call__(self, idx):
        return self.labels[idx].item()

class WILDSOracle(Oracle):

    def __init__(self, dataset, task=None, base_dir="../wilds/data/", 
                 loss_fn=None, accuracy_fn=None, device='cuda', **kwargs):
        
        from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
        from wilds.datasets.iwildcam_dataset import IWildCamDataset
        from wilds.datasets.amazon_dataset import AmazonDataset
        from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
        from wilds.datasets.fmow_dataset import FMoWDataset
        from wilds.datasets.rxrx1_dataset import RxRx1Dataset
        wilds_datasets = {
            'iwildcam': IWildCamDataset,
            'camelyon': Camelyon17Dataset,
            'amazon': AmazonDataset,
            'civilcomments': CivilCommentsDataset,
            'fmow': FMoWDataset,
            'rxrx1': RxRx1Dataset
        }

        dataset = wilds_datasets[task](root_dir=base_dir)
        dataset.get_subset('test').__dict__
        y_test = dataset.y_array[dataset.get_subset('test').indices].to(torch.device(device))
        self.labels = y_test
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn

class MODELSELECTOROracle(Oracle):

    def __init__(self, dataset, task=None, base_dir="../model-selector/resources/datasets/",
                 loss_fn=None, accuracy_fn=None, device='cuda', **kwargs):
        self.labels = torch.tensor(np.load(f'{base_dir}/{task}/oracle.npy'), 
                                   device=device).squeeze()
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn