class Oracle:
    def __init__(self, dataset, loss_fn=None):
        self.dataset = dataset
        self.loss_fn = loss_fn
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
        return self.loss_fn(preds.reshape(-1, C), self.labels.repeat(H), 
                            reduction='none').view(H, N).mean(dim=1)

    def __call__(self, idx):
        return self.labels[idx].item()