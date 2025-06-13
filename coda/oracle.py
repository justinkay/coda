import torch


class Oracle:
    def __init__(self, dataset, loss_fn=None, accuracy_fn=None):
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.device = dataset.device
        self.labels = dataset.labels
        assert self.labels is not None, "Oracle needs labels!"

    def true_losses(self, preds):
        """
        Compute the mean loss for each model.
        
        Args:
        - preds: Tensor of shape (H, N, C) representing post-softmax scores from each model for each data point.
        
        Returns:
        - Tensor of shape (H,) representing the mean loss for each model.
        """
        H, N, C = preds.shape
        return self.loss_fn(preds.view(-1, C), self.labels.repeat(H), 
                            reduction='none').view(H, N).mean(dim=1)

    def true_accuracies(self, preds):
        H, N, C = preds.shape
        accuracies = []
        batch_size = 100  # Adjust this value based on your GPU memory
        for i in range(0, H, batch_size):
            batch_end = min(i + batch_size, H)
            batch_preds = preds[i:batch_end]
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