class Ensemble:
    def __init__(self, preds, **kwargs):
        self.preds = preds
        self.device = preds.device
        H, N, C = preds.shape

    def get_preds(self, **kwargs):
        return self.preds.mean(dim=0)