import torch
import torch.nn.functional as F

from surrogates import losses, expected_error


class EmpiricalRisk:
    def __init__(self, dataset, loss_fn, **kwargs):
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.d_l_idxs = []
        self.d_l_ys = []

    def add_observation(self, d_idx, d_y):
        self.d_l_idxs.append(d_idx)
        self.d_l_ys.append(d_y)

    def get_risks_and_vars(self):
        risk = torch.zeros(self.dataset.pred_logits.shape[0], device=self.dataset.device)
        if len(self.d_l_idxs) > 0:
            for idx, label in zip(self.d_l_idxs, self.d_l_ys):
                risk += self.loss_fn(self.dataset.pred_logits[:,idx,:].squeeze(1), 
                                    torch.tensor([label], device=self.dataset.device).repeat(self.dataset.pred_logits.shape[0]))
            risk /= len(self.d_l_idxs)
        return risk, torch.zeros_like(risk) # TODO


class LUREEstimator:
    def __init__(self, H, N, C, loss_fn, device, **kwargs):
        """
        H: Number of models in the ensemble
        N: Total number of data points
        C: Number of classes
        loss_fn: Loss function to compute losses between predictions and true labels
        device: The device (CPU/GPU) to perform computations
        """
        self.H = H
        self.N = N
        self.C = C
        self.loss_fn = loss_fn
        self.device = device

        # Actively sampled points
        self.M = 0  # Number of sampled points
        self.losses = []  # True losses for each model - shape (H, M)
        self.qs = []  # Sampling probabilities for each point - shape (M,)

    def get_vs(self):
        """
        Compute the LURE weights (v_m) for each sampled point based on the current state.
        This is called after adding a new observation to update the weights.
        """
        vs = []
        for m in range(self.M):
            q = self.qs[m]
            m = m + 1 # M is 1-indexed when computing v
            v = 1 + ( (self.N - self.M) / (self.N - m) ) * (1 / ((self.N - m + 1) * q) - 1)
            vs.append(v)
        return vs

    def add_observation(self, pred_logits, label, qm, **kwargs):
        """
        Adds a new observation with the predicted logits, true label, and sampling probability.

        Args:
            pred_logits: shape (H, C) - predicted logits for each model
            label: int - true label for the sampled point
            qm: float - sampling probability of this point
        """
        loss = self.loss_fn(pred_logits, torch.tensor([label], device=self.device).repeat(self.H), reduction='none')
        self.losses.append(loss)
        self.qs.append(qm)
        self.M += 1

    def get_risks_and_vars(self):
        # Stack losses to shape (H, M)
        losses = torch.stack(self.losses, dim=1).view(self.H, -1)  # Shape (H, M)
        vs = torch.tensor(self.get_vs(), device=self.device).unsqueeze(0)  # Shape (1, M)

        # Compute weighted losses: w_m = v_m * L_m
        weighted_losses = vs * losses  # Shape (H, M)
        # TESTING: IGNORE VS
        # weighted_losses = losses

        # Compute LURE estimates (mean of weighted losses)
        lure_estimates = weighted_losses.mean(dim=1)  # Shape (H,)

        # Compute sample variance of weighted losses (Var[w_m])
        sample_variance = weighted_losses.var(dim=1, unbiased=True)  # Shape (H,)

        # Compute variance of the LURE estimator (Var[hat{R}_{LURE}] = Var[w_m] / M)
        variance_lure = sample_variance / self.M  # Shape (H,)

        # print("self losses stack [0]", torch.stack(self.losses, dim=1)[0].shape, torch.stack(self.losses, dim=1)[0])
        # print("vs", vs)
        # print("weighted losses 0", weighted_losses[0])

        return lure_estimates, variance_lure


class ASEEstimator:
    def __init__(self, dataset, surrogate, use_expected_loss=False):
        self.dataset = dataset
        self.surrogate = surrogate
        self.use_expected_loss = use_expected_loss
        self.d_l_idxs = []
        self.d_l_ys = []

    def get_risks_and_vars(self, loss_fn): # TODO figure out interface for loss_fn
        self.surrogate.retrain(d_l_idxs=self.d_l_idxs, d_l_ys=self.d_l_ys)
        if self.use_expected_loss:
            losss = expected_error(self.surrogate, self.dataset.pred_logits).mean(dim=-1)
        else:
            losss = losses(self.surrogate, self.dataset.pred_logits, loss_fn).mean(dim=-1)
        vars = torch.zeros(losss.shape[0]) # TODO?
        return losss, vars
    
    def add_observation(self, d_idx, d_y):
        self.d_l_idxs.append(d_idx)
        self.d_l_ys.append(d_y)


class ASEPPIEstimator:
    def __init__(self, dataset, surrogate, use_expected_loss=False, use_model_scores=True):
        self.dataset = dataset
        self.surrogate = surrogate
        self.d_l_idxs = []
        self.d_l_ys = []
        self.use_expected_loss = use_expected_loss
        self.use_model_scores = use_model_scores

    def get_risks_and_vars(self):
        # self.surrogate.retrain(d_l_idxs=self.d_l_idxs, d_l_ys=self.d_l_ys, epochs=50)
        surr_scores = self.surrogate.get_preds() # (N, C)
        model_scores = torch.softmax(self.dataset.pred_logits, dim=-1) # (H, N, C)

        if self.use_expected_loss:
            surr_losses = expected_error(self.surrogate, self.dataset.pred_logits) # (H, N)
        else:
            surr_losses = losses(self.surrogate, self.dataset.pred_logits, self.loss_fn)
        
        ppi_term1 = surr_losses.mean(dim=-1) # (H,)
        ppi_term2 = torch.zeros_like(ppi_term1)  # (H,)

        if len(self.d_l_idxs) > 0:
            for idx, label in zip (self.d_l_idxs, self.d_l_ys):
                # predicted risks for each model for this data point
                # either expected loss (if self.use_expected_loss) or 0/1 loss
                surr_predicted_risk = surr_losses[:, idx]

                # actual risks for each model for this data point
                # based either on model score for the true class (if self.use_model_scores) or 0/1 loss
                if self.use_model_scores:
                    model_true_risk = (1 - model_scores[:, idx, label])  # (H, 1, 1) # or should this be 0/1?
                else:
                    model_true_risk = 1 - (torch.argmax(model_scores[:, idx, :], dim=-1) == torch.tensor([label]).repeat(model_scores.shape[0]))
                
                diff = model_true_risk - surr_predicted_risk
                print("surr risk of true class",   surr_predicted_risk[:5])
                print("model risk of true class", model_true_risk[:5])
                print("diff[:5]", diff[:5])
                ppi_term2 += diff
            
            ppi_term2 /= len(self.d_l_idxs)
            print("ppi_term1[;5]",ppi_term1[:5])
            print("ppi_term2[:5]",ppi_term2[:5])

        ppi_estimate = ppi_term1 + ppi_term2

        vars = torch.zeros_like(ppi_estimate) # TODO
        return ppi_estimate, vars

    def add_observation(self, d_idx, d_y):
        self.d_l_idxs.append(d_idx)
        self.d_l_ys.append(d_y)


class ASIEstimator:

    def __init__(self, H, N, C, loss_fn, device, surrogate):
        """
        H: Number of models in the ensemble
        N: Total number of data points
        C: Number of classes
        loss_fn: Loss function to compute losses between predictions and true labels
        device: The device (CPU/GPU) to perform computations
        """
        self.H = H
        self.N = N
        self.C = C
        self.loss_fn = loss_fn
        self.device = device
        self.surrogate = surrogate

        # Actively sampled points
        self.M = 0  # Number of sampled points
        self.M_idxs = [] # Indices of sampled points
        self.losses = []  # True losses for each model - shape (H, M)
        self.pis = []  # Sampling probabilities for each point - shape (M,)
        self.labels = []

    def add_observation(self, pred_logits, label, pi_m, d_m_idx):
        """
        Adds a new observation with the predicted logits, true label, and sampling probability.

        Args:
            pred_logits: shape (H, C) - predicted logits for each model
            label: int - true label for the sampled point
            pi_m: float - sampling probability of this point
        """
        print("in add obs, pred_logits", pred_logits.shape)
        loss = self.loss_fn(pred_logits, torch.tensor([label], device=self.device).repeat(self.H), reduction='none') # shape (H,)
        print("in add obs, loss", loss.shape)
        self.losses.append(loss)

        # this is kind of a hack -- we assume an 'imaginary budget' equal to this time step
        # which means that each pi_m should be multiplied by this budget
        # since they otherwise sum to 1
        self.pis.append(pi_m*(self.M + 1))

        self.M += 1
        self.M_idxs.append(d_m_idx)
        self.labels.append(label)

    # TODO copied from selection.py _losses()
    def surrogate_losses(self):
        """
        Return:
            losses: shape (H, N) -- estimated losses for each hypothesis for each data point
        """
        pred_logits = self.surrogate.pred_logits
        H, N, C = pred_logits.shape

        # this is assumed to be softmaxed already
        surrogate_preds_tensor = self.surrogate.get_preds() # (weights=self.last_p_h)
        print("sum of first surr pred - should be 1", surrogate_preds_tensor[0,:].sum())

        batch_size = 1000
        losses = torch.zeros(H, N, device=self.device)
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            batch_preds = pred_logits[:, i:batch_end, :].reshape(-1, C)
            batch_surrogate = surrogate_preds_tensor[i:batch_end].repeat(H, 1) # ground truth on these points is same for every model
            batch_loss = self.loss_fn( 
                batch_preds,
                torch.argmax(batch_surrogate, dim=-1), # TODO: this only works with argmax; not with raw or softmax'd probs
                reduction='none'
            ).view(H, -1)

            losses[:, i:batch_end] = batch_loss

        return losses

    def get_risks_and_vars(self):
        # Step 1: Convert stored data to tensors
        true_losses = torch.stack(self.losses, dim=1).to(self.device)  # (H, M)
        pis_tensor = torch.tensor(self.pis, device=self.device).unsqueeze(0)  # (1, M)

        # Step 2: Calculate surrogate risks
        surrogate_preds_tensor = self.surrogate.get_preds()
        confidence_scores, pseudo_labels = surrogate_preds_tensor.max(dim=-1)
        surrogate_risk = self.surrogate_losses()  # (H, N)
        # reweight by confidence
        surrogate_risk /= confidence_scores
        risk_means = surrogate_risk.mean(dim=1)  # (H,)

        # Step 3: Compute adjustment terms
        adjustment_samples = (true_losses - surrogate_risk[:, self.M_idxs]) / pis_tensor  # (H, M)
        adjustment_terms = adjustment_samples.sum(dim=1)  # (H,)
        risk_means += adjustment_terms / self.N

        # Step 4: Compute variance
        sample_variance = torch.var(adjustment_samples, dim=1, unbiased=True)  # (H,)
        risk_vars = (sample_variance * self.M) / (self.N ** 2)  # (H,)
        risk_vars = torch.clamp(risk_vars, min=0)

        print("true losses of labeled points (first 4 models)", true_losses[:4, :])
        print("surrogate losses of labeled points (first 4 models)", surrogate_risk[:4, self.M_idxs])
        print("pis tensor", pis_tensor)
        if len(self.M_idxs) == 2:
            print("how did this happen?")
            print("true losses", true_losses[:4, 1])
            print("surrogate losses", surrogate_risk[:4, self.M_idxs[1]])
            print("true label", self.labels[1])

            surrogate_preds_tensor = self.surrogate.get_preds()
            print("surrogate pred labels", torch.argmax(surrogate_preds_tensor[self.M_idxs[1], :], dim=-1))
            print("surrogate pred max prob", surrogate_preds_tensor[self.M_idxs[1], :].max(dim=-1))

            print("model pred labels", torch.argmax(self.surrogate.pred_logits[:4, self.M_idxs[1], :], dim=-1))
            print("model pred max prob", torch.softmax(self.surrogate.pred_logits[:4, self.M_idxs[1], :], dim=-1).max(dim=-1))

        return risk_means, risk_vars

class BetaBinomialEstimator:
    def __init__(self, accuracy_priors, accuracy_prior_strength=10.0, item_priors=None):
        """
        accuracy_priors: shape (H,), initial accuracies for each model, b/w 0 and 1
        item_priors: shape (N,C), class priors for each item
        """
        self.alpha = accuracy_priors * accuracy_prior_strength
        self.beta  = (1.0 - accuracy_priors) * accuracy_prior_strength
        self.p_item = item_priors

    def add_observation(self, 
                        pred_logits, # (H, C)
                        d_m_idx, d_m_y):
        pred_cls = torch.argmax(pred_logits, dim=-1)
        