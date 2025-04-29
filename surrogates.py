import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from typing import Optional, Generator, Dict


def losses(surrogate, pred_logits, loss_fn):
    """
    Get losses of some other predictions (pred_logits) compared to this surrogate.
    Could probably remove the batching.
    """
    H, N, C = pred_logits.shape
    surrogate_probs = surrogate.get_preds() # Shape: (N, C)
    surrogate_labels = torch.argmax(surrogate_probs, dim=-1)  # Shape: (N,)

    batch_size = 1000
    losses = torch.zeros(H, N, device=pred_logits.device)
    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        batch_preds = pred_logits[:, i:batch_end, :]  # Shape: (H, batch_size, C)
        batch_preds = batch_preds.reshape(-1, C)  # Shape: (H * batch_size, C)
        batch_surrogate_labels = surrogate_labels[i:batch_end].repeat(H)  # Shape: (H * batch_size,)

        batch_loss = loss_fn(
            batch_preds,
            batch_surrogate_labels.long()
        ).view(H, -1)

        losses[:, i:batch_end] = batch_loss

    return losses

def expected_error(surrogate, pred_logits):
    """
    Get expected error (``accuracy losses'') of some other predictions (pred_logits) compared to this surrogate.
    """
    H, N, C = pred_logits.shape
    pi_y = surrogate.get_preds() # Shape: (N, C)

    pred_probs = torch.softmax(pred_logits, dim=-1)
    pred_classes = pred_probs.argmax(dim=-1)

    y_star_probs = pi_y[torch.arange(pi_y.shape[0]), pred_classes]
    exp_accuracy_loss = 1 - y_star_probs

    return exp_accuracy_loss


class Ensemble:
    def __init__(self, pred_logits, **kwargs):
        self.pred_logits = pred_logits
        self.device = pred_logits.device
        H, N, C = pred_logits.shape

    def get_preds(self, **kwargs):
        return torch.softmax(self.pred_logits, dim=-1).mean(dim=0)


class WeightedEnsemble(Ensemble):
    def __init__(self, pred_logits, **kwargs):
        super().__init__(pred_logits)

    def get_preds(self, weights):
        return (torch.softmax(self.pred_logits, dim=-1) * weights.view(-1, 1, 1)).sum(dim=0)


class OracleSurrogate:
    def __init__(self, oracle, **kwargs):
        self.oracle = oracle
        self.device = oracle.device
        self.pred_logits = self.oracle.dataset.pred_logits # this is dumb

    def get_preds(self, **kwargs):
        H,N,C = self.oracle.dataset.pred_logits.shape
        labels = self.oracle.labels
        one_hot = torch.nn.functional.one_hot(labels, num_classes=C).float()
        return one_hot


# things that didn't help:
# lasso regularization
# per-model-per-class weights (H,C)
# removing per-class bias
# replacing softmax on weights with sigmoid
class TrainableEnsemble(Ensemble):
    def __init__(self, pred_logits, train_to_convergence=False):
        super().__init__(pred_logits)
        self.pred_logits = pred_logits
        self.init_weights_and_biases()
        self.train_to_convergence = train_to_convergence

    def init_weights_and_biases(self):
        H, N, C = self.pred_logits.shape
        device = self.pred_logits.device
        
        # per-model weights
        self.model_weight_logits = torch.nn.Parameter(torch.zeros(H, device=device), requires_grad=True)  # Logits for weights
        self.bias = torch.nn.Parameter(torch.zeros(C, device=device), requires_grad=True)    # Bias per class
        self.temperature = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=True)
        self.optimizer = optim.Adam([self.model_weight_logits, self.bias, self.temperature], lr=0.01, weight_decay=1e-4)

    def retrain(self, d_l_idxs, d_l_ys):
        if self.train_to_convergence:
            min_epochs = 0
            max_epochs = 1000
        else:
            min_epochs = max_epochs =50
            
        device = self.pred_logits.device
        self.init_weights_and_biases()

        logits_stack = self.pred_logits[:, d_l_idxs, :]  # Shape: (H, M, C)
        # print("logits_stack data point 0", logits_stack[:, 0, :])

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(max_epochs):
            # Compute per-model weights using softmax over logits
            weights = torch.softmax(self.model_weight_logits, dim=0)  # Shape: (H,)
            # print("weights", weights.shape, weights)

            # Weight each model in the ensemble
            # We do this in logit space so we can softmax later
            weighted_logits = torch.einsum('h, hmc -> mc', weights, logits_stack)  # Shape: (M, C)
            # print("weighted_logits", weighted_logits.shape, weighted_logits)

            # Add bias term (broadcasted over (M, C))
            scaled_logits = weighted_logits / self.temperature
            weighted_logits_with_bias = scaled_logits + self.bias  # Shape: (M, C)
            # print("weighted_logits_with_bias", weighted_logits_with_bias.shape, weighted_logits_with_bias)

            # Cross entropy with labeled data points
            loss = criterion(weighted_logits_with_bias, torch.tensor(d_l_ys, device=device).long())
            # print("loss", loss.shape, loss)

            self.optimizer.zero_grad()
            loss.backward()

            # print("Gradients for logits:", self.logits.grad)
            # print("Gradient for bias:", self.bias.grad)

            self.optimizer.step()

            with torch.no_grad():
                preds = self.get_preds()[d_l_idxs, :]
                predicted_labels = torch.argmax(preds, dim=-1)  # Shape: (N,)
                accuracy = (predicted_labels == torch.tensor(d_l_ys, device=device)).sum().item() / len(d_l_ys)
                if epoch > min_epochs and accuracy > 0.99:
                    break
        
        print("Accuracy at epoch", epoch, "is", accuracy)

        # print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        # print("Weights:", weights.detach().numpy())
        # print("Bias:", self.bias.detach().numpy())

    def get_preds(self):
        """
        Get (softmaxed) predictions of the weighted ensemble across the entire dataset.
        """
        logits = self.pred_logits  # Shape: (H, N, C)
        weights = torch.softmax(self.model_weight_logits, dim=0)  # Shape: (H,)
        weighted_logits = torch.einsum('h, hnc -> nc', weights, logits)  # Shape: (N, C)
        scaled_logits = weighted_logits / self.temperature
        weighted_logits_with_bias = scaled_logits + self.bias  # Shape: (N, C)
        probs = torch.softmax(weighted_logits_with_bias, dim=-1)  # Shape: (N, C)
        return probs


class EMsemble:
    def __init__(self, pred_logits: torch.Tensor, 
                 **kwargs):
        """
        Initialize the EMEnsemble with predicted logits from multiple models.

        Args:
            pred_logits (torch.Tensor): Tensor of shape (H, N, C),
                                        where H is the number of models,
                                        N is the number of data points,
                                        and C is the number of classes.
            device (str, optional): Device to perform computations on.
                                    Defaults to 'cpu'.
        """
        assert pred_logits.ndim == 3, "pred_logits must be a 3D tensor (H, N, C)"
        self.H, self.N, self.C = pred_logits.shape
        self.device = pred_logits.device
        self.pred_logits = pred_logits.to(self.device)  # Shape: (H, N, C)
        
        # Initialize logits for weights (unconstrained), requires_grad=True
        self.weights_logits = torch.zeros(self.H, device=self.device, requires_grad=True)

    # def fit_iter(
    #     self,
    #     max_iter: int = 100,
    #     tol: float = 1e-4,
    #     lr: float = 0.1,
    #     verbose: bool = False,
    #     entropy_weight: float = 0.1,
    # ) -> Generator[Dict[str, float], None, None]:
    #     """
    #     Generator-based method to fit the EM ensemble, yielding information after each iteration.

    #     Args:
    #         max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
    #         tol (float, optional): Tolerance for convergence based on weight changes. Defaults to 1e-4.
    #         lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
    #         verbose (bool, optional): If True, prints progress during fitting. Defaults to False.
    #         entropy_weight (float, optional): Weight for entropy regularization. Defaults to 0.1.
    #         class_weights (torch.Tensor, optional): Weights for each class to handle imbalance.

    #     Yields:
    #         Generator[Dict[str, float], None, None]: A dictionary containing iteration info.
    #     """
    #     H, N, C = self.pred_logits.shape
    #     optimizer = optim.Adam([self.weights_logits], lr=lr)
    #     previous_weights = F.softmax(self.weights_logits.detach(), dim=0).clone()

    #     for iteration in range(1, max_iter + 1):
    #         optimizer.zero_grad()

    #         # # attempt 1
    #         # # E-Step: Compute ensemble probabilities (soft pseudolabels)
    #         # weights_normalized = F.softmax(self.weights_logits, dim=0)  # w_h   # Shape: (H,)
    #         # weighted_probs = weights_normalized.view(-1, 1, 1) * F.softmax(self.pred_logits, dim=-1) # w_h * P_h(c|n)   # Shape: (H, N, C)
    #         # ensemble_probs = weighted_probs.sum(dim=0)  # P(c|n) = Σ w_h P_h(c|n)   # Shape: (N, C)

    #         # # M-Step: Optimize weights to maximize expected log-likelihood
    #         # log_probs = torch.log(weighted_probs + 1e-12) # log(w_h P_h(c|n))    # Shape: (H, N, C)

    #         # # Combine labeled and unlabeled parts by up-weighting the labeled component
    #         # total_log_likelihood = (ensemble_probs.unsqueeze(0) * log_probs).sum(dim=(1, 2)) # Σ P(c|n) log(w_h P_h(c|n))     # Shape: (H,)
    #         # loss1 = -total_log_likelihood.mean() # Negative for minimization


    #         # attempt 2....
    #         # E-Step: Compute ensemble probabilities (soft pseudolabels)
    #         weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H,)
    #         P_h = F.softmax(self.pred_logits, dim=-1)  # Shape: (H, N, C)
    #         ensemble_probs = (weights_normalized.view(-1, 1, 1) * P_h).sum(dim=0)  # Shape: (N, C) # exactly the same as before so far

    #         # M-Step: Compute the expected complete-data log-likelihood
    #         log_P_h = torch.log(P_h + 1e-12)  # Shape: (H, N, C)
    #         log_weights = torch.log(weights_normalized + 1e-12).view(-1, 1, 1)  # Shape: (H, 1, 1)

    #         # Compute weighted log probabilities
    #         weighted_log_probs = (log_weights + log_P_h)  # Shape: (H, N, C)

    #         # Compute loss
    #         loss2 = -(ensemble_probs * (weights_normalized.view(-1, 1, 1) * weighted_log_probs).sum(dim=0)).sum()

    #         loss = loss2



    #         # Entropy Regularization across all weights
    #         entropy = -(weights_normalized * torch.log(weights_normalized + 1e-12)).sum()
    #         loss -= entropy_weight * entropy  # Subtract to encourage high entropy


    #         # attempt 3.....
    #         # totally diverges
    #         # Compute model probabilities
    #         # P_h = F.softmax(self.pred_logits, dim=-1)  # Shape: (H, N, C)

    #         # # Compute weights
    #         # weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H,)
    #         # weights_expanded = weights_normalized.view(-1, 1, 1)  # Shape: (H, 1, 1)

    #         # # Compute ensemble probabilities
    #         # ensemble_probs = (weights_expanded * P_h).sum(dim=0)  # Shape: (N, C)

    #         # # Compute responsibilities
    #         # gamma = (weights_expanded * P_h) / (ensemble_probs + 1e-12)  # Shape: (H, N, C)

    #         # # Compute log probabilities
    #         # log_w = torch.log(weights_normalized + 1e-12).view(-1, 1, 1)  # Shape: (H, 1, 1)
    #         # log_P_h = torch.log(P_h + 1e-12)  # Shape: (H, N, C)

    #         # # Compute expected log-likelihood
    #         # expected_log_likelihood = gamma * (log_w + log_P_h)  # Shape: (H, N, C)
    #         # loss = -expected_log_likelihood.sum()

    #         # # Entropy Regularization
    #         # entropy = -(weights_normalized * torch.log(weights_normalized + 1e-12)).sum()
    #         # loss -= entropy_weight * entropy  # Encourage high entropy


    #         # Backpropagation
    #         loss.backward()
    #         optimizer.step()

    #         # Compute weight changes for convergence check
    #         current_weights = F.softmax(self.weights_logits.detach(), dim=0)
    #         weight_change = torch.max(torch.abs(current_weights - previous_weights)).item()

    #         # Prepare information to yield
    #         info = {
    #             'iteration': iteration,
    #             'loss': loss.item(),
    #             'entropy': entropy.item(),
    #             'weight_change': weight_change,
    #             'weights': current_weights.cpu().numpy()
    #         }

    #         if verbose:
    #             print(f"Iteration {iteration}: Loss={info['loss']:.6f}, "
    #                   f"Entropy={info['entropy']:.6f}, "
    #                 #   f"Weight Change={info['weight_change']:.6f}, "
    #                 #   f"Weights={info['weights']}"
    #                   )

    #         # Yield iteration info
    #         yield info

    #         # Check for convergence
    #         if weight_change < tol:
    #             if verbose:
    #                 print(f"Converged at iteration {iteration} with loss {info['loss']}")
    #             break

    #         # Update previous_weights for next iteration
    #         previous_weights = current_weights.clone()

    # def retrain(
    # # def fit(
    #     self,
    #     max_iter: int = 100,
    #     tol: float = 1e-4,
    #     lr: float = 0.1,
    #     verbose: bool = True,
    #     entropy_weight: float = 0.1,
    # ):
    #     """
    #     Traditional fit method that runs the fitting process to completion by iterating through fit_iter().

    #     Args:
    #         max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
    #         tol (float, optional): Tolerance for convergence based on weight changes. Defaults to 1e-4.
    #         lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
    #         verbose (bool, optional): If True, prints progress during fitting. Defaults to False.
    #         entropy_weight (float, optional): Weight for entropy regularization. Defaults to 0.1.
    #         class_weights (torch.Tensor, optional): Weights for each class to handle imbalance.
    #     """
    #     for _ in self.fit_iter(
    #         max_iter=max_iter,
    #         tol=tol,
    #         lr=lr,
    #         verbose=verbose,
    #         entropy_weight=entropy_weight,
    #     ):
    #         pass  # The fitting process is handled within the generator

    # def get_preds(self) -> torch.Tensor:
    #     """
    #     Compute the ensemble probabilities using the optimized weights.

    #     Returns:
    #         torch.Tensor: Tensor of shape (N, C) containing the ensemble probabilities.
    #     """
    #     with torch.no_grad():
    #         weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H,)
    #         weighted_probs = weights_normalized.view(-1, 1, 1) * F.softmax(self.pred_logits, dim=-1)  # Shape: (H, N, C)
    #         ensemble_probs = weighted_probs.sum(dim=0)  # Shape: (N, C)
    #         ensemble_probs = F.normalize(ensemble_probs, p=1, dim=-1)  # Ensure probabilities sum to 1
    #     return ensemble_probs

    def fit_iter(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        lr: float = 0.1,
        verbose: bool = False,
        entropy_weight: float = 0.1,
        batch_size_N: int = 1000,  # Added batch_size_N parameter
    ) -> Generator[Dict[str, float], None, None]:
        """
        Generator-based method to fit the EM ensemble, yielding information after each iteration.

        Args:
            max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
            tol (float, optional): Tolerance for convergence based on weight changes. Defaults to 1e-4.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            verbose (bool, optional): If True, prints progress during fitting. Defaults to False.
            entropy_weight (float, optional): Weight for entropy regularization. Defaults to 0.1.
            batch_size_N (int, optional): Batch size for processing data points to reduce memory usage.

        Yields:
            Generator[Dict[str, float], None, None]: A dictionary containing iteration info.
        """
        H, N, C = self.pred_logits.shape
        optimizer = optim.Adam([self.weights_logits], lr=lr)
        previous_weights = F.softmax(self.weights_logits.detach(), dim=0).clone()

        for iteration in range(1, max_iter + 1):
            optimizer.zero_grad()

            weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H,)
            log_weights = torch.log(weights_normalized + 1e-12).view(-1, 1, 1)  # Shape: (H, 1, 1)

            # Initialize total loss for this iteration
            total_loss = 0.0

            # Process data in batches along the N dimension
            for start in range(0, N, batch_size_N):
                end = min(start + batch_size_N, N)
                pred_logits_batch = self.pred_logits[:, start:end, :]  # Shape: (H, batch_size_N, C)

                # E-Step: Compute ensemble probabilities (soft pseudolabels)
                P_h_batch = F.softmax(pred_logits_batch, dim=-1)  # Shape: (H, batch_size_N, C)
                ensemble_probs_batch = (weights_normalized.view(-1, 1, 1) * P_h_batch).sum(dim=0)  # Shape: (batch_size_N, C)

                # M-Step: Compute the expected complete-data log-likelihood
                log_P_h_batch = torch.log(P_h_batch + 1e-12)  # Shape: (H, batch_size_N, C)
                weighted_log_probs_batch = log_weights + log_P_h_batch  # Shape: (H, batch_size_N, C)

                # Compute loss for the current batch
                term = (weights_normalized.view(-1, 1, 1) * weighted_log_probs_batch).sum(dim=0)  # Shape: (batch_size_N, C)
                loss_batch = -(ensemble_probs_batch * term).sum()  # Scalar
                total_loss += loss_batch  # Accumulate loss over batches

            # Entropy Regularization across all weights
            entropy = -(weights_normalized * torch.log(weights_normalized + 1e-12)).sum()
            loss = total_loss - entropy_weight * entropy  # Total loss for this iteration

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Compute weight changes for convergence check
            current_weights = F.softmax(self.weights_logits.detach(), dim=0)
            weight_change = torch.max(torch.abs(current_weights - previous_weights)).item()

            # Prepare information to yield
            info = {
                'iteration': iteration,
                'loss': loss.item(),
                'entropy': entropy.item(),
                'weight_change': weight_change,
                'weights': current_weights.cpu().numpy()
            }

            if verbose:
                print(f"Iteration {iteration}: Loss={info['loss']:.6f}, "
                      f"Entropy={info['entropy']:.6f}, "
                      f"Weight Change={info['weight_change']:.6f}")

            # Yield iteration info
            yield info

            # Check for convergence
            if weight_change < tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with loss {info['loss']}")
                break

            # Update previous_weights for next iteration
            previous_weights = current_weights.clone()

    def retrain(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        lr: float = 0.1,
        verbose: bool = True,
        entropy_weight: float = 0.1,
        batch_size_N: int = 1000,  # Added batch_size_N parameter
    ):
        """
        Traditional fit method that runs the fitting process to completion by iterating through fit_iter().

        Args:
            max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
            tol (float, optional): Tolerance for convergence based on weight changes. Defaults to 1e-4.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            verbose (bool, optional): If True, prints progress during fitting. Defaults to False.
            entropy_weight (float, optional): Weight for entropy regularization. Defaults to 0.1.
            batch_size_N (int, optional): Batch size for processing data points to reduce memory usage.
        """
        for _ in self.fit_iter(
            max_iter=max_iter,
            tol=tol,
            lr=lr,
            verbose=verbose,
            entropy_weight=entropy_weight,
            batch_size_N=batch_size_N,
        ):
            pass  # The fitting process is handled within the generator

    def get_preds(self, batch_size_N: int = 1000) -> torch.Tensor:
        """
        Compute the ensemble probabilities using the optimized weights.

        Args:
            batch_size_N (int, optional): Batch size for processing data points to reduce memory usage.

        Returns:
            torch.Tensor: Tensor of shape (N, C) containing the ensemble probabilities.
        """
        with torch.no_grad():
            weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H,)
            N = self.N
            ensemble_probs_list = []

            # Process data in batches along the N dimension
            for start in range(0, N, batch_size_N):
                end = min(start + batch_size_N, N)
                pred_logits_batch = self.pred_logits[:, start:end, :]  # Shape: (H, batch_size_N, C)
                P_h_batch = F.softmax(pred_logits_batch, dim=-1)  # Shape: (H, batch_size_N, C)
                weighted_probs_batch = weights_normalized.view(-1, 1, 1) * P_h_batch  # Shape: (H, batch_size_N, C)
                ensemble_probs_batch = weighted_probs_batch.sum(dim=0)  # Shape: (batch_size_N, C)
                ensemble_probs_list.append(ensemble_probs_batch)

            # Concatenate all batches to get the final ensemble probabilities
            ensemble_probs = torch.cat(ensemble_probs_list, dim=0)  # Shape: (N, C)
            ensemble_probs = F.normalize(ensemble_probs, p=1, dim=-1)  # Ensure probabilities sum to 1

        return ensemble_probs

    def get_weights(self) -> torch.Tensor:
        """
        Get the optimized ensemble weights.

        Returns:
            torch.Tensor: Tensor of shape (H,) containing the optimized weights.
        """
        with torch.no_grad():
            return F.softmax(self.weights_logits, dim=0)
        

# class EMsemblePMPC(EMsemble):
#     """Per-model-per-class weights"""
#     def __init__(self, pred_logits: torch.Tensor, 
#                  d_l_weight = 1., # how much weight to give to labeled points (w.r.t. weight of 1 for unlabeled pts)
#                  **kwargs):
#         super().__init__(pred_logits=pred_logits, d_l_weight=d_l_weight, **kwargs)
#         self.weights_logits = torch.zeros(self.H, self.C, device=self.device, requires_grad=True)


#     def fit_iter(
#         self,
#         max_iter: int = 100,
#         tol: float = 1e-4,
#         lr: float = 0.1,
#         verbose: bool = False,
#         entropy_weight: float = 0.1,
#         d_l_idxs = [], d_l_ys = [], # if labeled points are available
#     ) -> Generator[Dict[str, float], None, None]:
#         """
#         Generator-based method to fit the EM ensemble, yielding information after each iteration.

#         Args:
#             max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
#             tol (float, optional): Tolerance for convergence based on weight changes. Defaults to 1e-4.
#             lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
#             verbose (bool, optional): If True, prints progress during fitting. Defaults to False.
#             entropy_weight (float, optional): Weight for entropy regularization. Defaults to 0.1.
#             class_weights (torch.Tensor, optional): Weights for each class to handle imbalance.

#         Yields:
#             Generator[Dict[str, float], None, None]: A dictionary containing iteration info.
#         """
#         H, N, C = self.pred_logits.shape
#         optimizer = optim.Adam([self.weights_logits], lr=lr)
#         previous_weights = F.softmax(self.weights_logits.detach(), dim=0).clone()

#         for iteration in range(1, max_iter + 1):
#             optimizer.zero_grad()

#             # # E-Step: Compute ensemble probabilities (soft pseudolabels)
#             # weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H, C)
#             # weighted_probs = weights_normalized.view(self.H, 1, self.C) * F.softmax(self.pred_logits, dim=-1)
#             # ensemble_probs = weighted_probs.sum(dim=0)  # Shape: (N, C)

#             # # M-Step: Optimize weights to maximize expected log-likelihood
#             # log_probs = torch.log(weighted_probs + 1e-12)  # Shape: (H, N, C)
#             # expected_log_likelihood = (ensemble_probs.unsqueeze(0) * log_probs).sum(dim=(1, 2))  # Shape: (H, C)
#             # loss = -expected_log_likelihood.mean()  # Negative for minimization

#             # E-Step: Compute ensemble probabilities (soft pseudolabels)
#             weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H,)
#             P_h = F.softmax(self.pred_logits, dim=-1)  # Shape: (H, N, C)
#             ensemble_probs = (weights_normalized.view(-1, 1, 1) * P_h).sum(dim=0)  # Shape: (N, C)

#             # M-Step: Compute the expected complete-data log-likelihood
#             log_P_h = torch.log(P_h + 1e-12)  # Shape: (H, N, C)
#             log_weights = torch.log(weights_normalized + 1e-12).view(-1, 1, 1)  # Shape: (H, 1, 1)

#             # Compute weighted log probabilities
#             weighted_log_probs = (log_weights + log_P_h)  # Shape: (H, N, C)

#             # Compute loss
#             loss = - (ensemble_probs * (weights_normalized.view(-1, 1, 1) * weighted_log_probs).sum(dim=0)).sum()

#             # Entropy Regularization across all weights
#             entropy = -(weights_normalized * torch.log(weights_normalized + 1e-12)).sum()
#             loss -= entropy_weight * entropy  # Subtract to encourage high entropy

#             # Backpropagation
#             loss.backward()
#             optimizer.step()

#             # Compute weight changes for convergence check
#             current_weights = F.softmax(self.weights_logits.detach(), dim=0)
#             weight_change = torch.max(torch.abs(current_weights - previous_weights)).item()

#             # Prepare information to yield
#             info = {
#                 'iteration': iteration,
#                 'loss': loss.item(),
#                 'entropy': entropy.item(),
#                 'weight_change': weight_change,
#                 'weights': current_weights.cpu().numpy()
#             }

#             if verbose:
#                 print(f"Iteration {iteration}: Loss={info['loss']:.6f}, "
#                       f"Entropy={info['entropy']:.6f}, "
#                       f"Weight Change={info['weight_change']:.6f}, "
#                     #   f"Weights={info['weights']}"
#                       )

#             # Yield iteration info
#             yield info

#             # Check for convergence
#             if weight_change < tol:
#                 if verbose:
#                     print(f"Converged at iteration {iteration} with loss {info['loss']}")
#                 break

#             # Update previous_weights for next iteration
#             previous_weights = current_weights.clone()


#     def get_preds(self) -> torch.Tensor:
#         """
#         Compute the ensemble probabilities using the optimized per-class weights.

#         Returns:
#             torch.Tensor: Tensor of shape (N, C) containing the ensemble probabilities.
#         """
#         with torch.no_grad():
#             weights_normalized = F.softmax(self.weights_logits, dim=0)  # Shape: (H, C)
#             weighted_probs = weights_normalized.view(self.H, 1, self.C) * F.softmax(self.pred_logits, dim=-1)
#             ensemble_probs = weighted_probs.sum(dim=0)  # Shape: (N, C)
#         return ensemble_probs

def simple_expected_error(pi_y, pred_logits):
    """
    Get expected error (``accuracy losses'') of some other predictions (pred_logits) compared to this surrogate.
    """
    H, N, C = pred_logits.shape
    # pi_y = surrogate.get_preds() # Shape: (N, C)

    pred_probs = torch.softmax(pred_logits, dim=-1)
    pred_classes = pred_probs.argmax(dim=-1)

    y_star_probs = pi_y[torch.arange(pi_y.shape[0]), pred_classes]
    exp_accuracy_loss = 1 - y_star_probs

    return exp_accuracy_loss

class PseudoEMsemble(EMsemble):
    """The 'EM step' just reweights by expected losses."""
    def __init__(self, pred_logits: torch.Tensor, 
                 **kwargs):
        """
        Initialize the EMEnsemble with predicted logits from multiple models.

        Args:
            pred_logits (torch.Tensor): Tensor of shape (H, N, C),
                                        where H is the number of models,
                                        N is the number of data points,
                                        and C is the number of classes.
            device (str, optional): Device to perform computations on.
                                    Defaults to 'cpu'.
        """
        assert pred_logits.ndim == 3, "pred_logits must be a 3D tensor (H, N, C)"
        self.H, self.N, self.C = pred_logits.shape
        self.device = pred_logits.device
        self.pred_logits = pred_logits.to(self.device)  # Shape: (H, N, C)
        
        self.weights = torch.ones(self.H, device=self.device) / self.H

    def fit_iter(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        lr: float = 0.1,
        verbose: bool = False,
        entropy_weight: float = 0.1,
    ) -> Generator[Dict[str, float], None, None]:
        H, N, C = self.pred_logits.shape
        previous_weights = self.weights.clone()

        for iteration in range(1, max_iter + 1):
            # ensemble predicted scores for each data point
            weights_normalized = self.weights #F.softmax(self.weights_logits, dim=0)  # w_h   # Shape: (H,)
            weighted_probs = weights_normalized.view(-1, 1, 1) * F.softmax(self.pred_logits, dim=-1) # w_h * P_h(c|n)   # Shape: (H, N, C)
            ensemble_probs = weighted_probs.sum(dim=0)  # P(c|n) = Σ w_h P_h(c|n)   # Shape: (N, C)

            # reweight by expected loss of each model
            expected_risk_means = simple_expected_error(ensemble_probs, self.pred_logits).mean(dim=-1)
            self.weights *= (1 - expected_risk_means) # TODO only works with accuracy
            self.weights /= self.weights.sum()
            
            entropy = -(weights_normalized * torch.log(weights_normalized + 1e-12)).sum()
            current_weights = self.weights.detach()
            weight_change = torch.max(torch.abs(current_weights - previous_weights)).item()

            # Prepare information to yield
            info = {
                'iteration': iteration,
                # 'loss': loss.item(),
                'entropy': entropy.item(),
                'weight_change': weight_change,
                'weights': current_weights.cpu().numpy()
            }

            if verbose:
                print(f"Iteration {iteration}:" # Loss={info['loss']:.6f}, "
                      f"Entropy={info['entropy']:.6f}, "
                      f"Weight Change={info['weight_change']:.6f}, "
                    #   f"Weights={info['weights']}"
                      )

            # Yield iteration info
            yield info

            # Check for convergence
            if weight_change < tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with loss {info['loss']}")
                break

            # Update previous_weights for next iteration
            previous_weights = current_weights.clone()

    def retrain(
    # def fit(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        lr: float = 0.1,
        verbose: bool = True,
        entropy_weight: float = 0.1,
    ):
        """
        Traditional fit method that runs the fitting process to completion by iterating through fit_iter().

        Args:
            max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
            tol (float, optional): Tolerance for convergence based on weight changes. Defaults to 1e-4.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            verbose (bool, optional): If True, prints progress during fitting. Defaults to False.
            entropy_weight (float, optional): Weight for entropy regularization. Defaults to 0.1.
            class_weights (torch.Tensor, optional): Weights for each class to handle imbalance.
        """
        for _ in self.fit_iter(
            max_iter=max_iter,
            tol=tol,
            lr=lr,
            verbose=verbose,
            entropy_weight=entropy_weight,
        ):
            pass  # The fitting process is handled within the generator

    def get_preds(self) -> torch.Tensor:
        """
        Compute the ensemble probabilities using the optimized weights.

        Returns:
            torch.Tensor: Tensor of shape (N, C) containing the ensemble probabilities.
        """
        with torch.no_grad():
            weights_normalized = self.weights
            weighted_probs = weights_normalized.view(-1, 1, 1) * F.softmax(self.pred_logits, dim=-1)  # Shape: (H, N, C)
            ensemble_probs = weighted_probs.sum(dim=0)  # Shape: (N, C)
            ensemble_probs = F.normalize(ensemble_probs, p=1, dim=-1)  # Ensure probabilities sum to 1
        return ensemble_probs

    def get_weights(self) -> torch.Tensor:
        """
        Get the optimized ensemble weights.

        Returns:
            torch.Tensor: Tensor of shape (H,) containing the optimized weights.
        """
        with torch.no_grad():
            return self.weights

# class DawidSkeneModel:
#     def __init__(self,
#                  class_num,
#                  max_iter=100,
#                  tolerance=0.01) -> None:
#         self.class_num = class_num
#         self.max_iter = max_iter
#         self.tolerance = tolerance

#     def run(self, dataset):
#         """
#         dataset: N, H, C (supposed to be 1-hot in original impl)
#         """
#         self.task_num, self.worker_num, _ = dataset.shape
#         self.dataset_tensor = dataset
#         predict_label =  self.dataset_tensor.sum(1) / self.dataset_tensor.sum(1).sum(1, keepdim=True)
#         # print(predict_label)

#         flag = True
#         prev_error_rates, prev_predict_label = None, None
#         iter_num = 0

#         while flag:
#             # print("predict_label",predict_label)
#             error_rates = self._m_step(predict_label)
#             next_predict_label = self._e_step(predict_label, error_rates)
#             log_L = self._get_likelihood(predict_label, error_rates)

#             if iter_num == 0:
#                 logging.info("{}\t{}".format(iter_num, log_L))
#             else:
#                 marginal_predict = torch.sum(predict_label, 0) / self.task_num
#                 prev_marginal_predict = torch.sum(prev_predict_label, 0) / self.task_num
#                 marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
#                 error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

#                 if self._check_condition(marginals_diff, error_rates_diff, iter_num):
#                     flag = False

#                 print("marginals_diff", marginals_diff, "error_rates_diff", error_rates_diff)

#             prev_error_rates = error_rates
#             prev_predict_label = predict_label
#             predict_label = next_predict_label
#             iter_num += 1

#         marginal_predict = torch.sum(predict_label, 0) / self.task_num
#         worker_reliability = {}
#         for i in range(self.worker_num):
#             ie_rates = marginal_predict * error_rates[i, :, :]
#             reliability = torch.sum(torch.diagonal(ie_rates))
#             worker_reliability[i] = reliability.item()
            
#         return marginal_predict, error_rates, worker_reliability, predict_label

#     def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
#         return (marginals_diff < self.tolerance and error_rates_diff < self.tolerance) or iter_num > self.max_iter

#     def _m_step(self, predict_label):
#         error_rates = torch.zeros((self.worker_num, self.class_num, self.class_num), dtype=torch.float32, device=predict_label.device)

#         # Equation 2.3
#         for i in range(self.class_num):
#             worker_error_rate = torch.einsum('t,twc->wc', predict_label[:, i], self.dataset_tensor)
#             sum_worker_error_rate = worker_error_rate.sum(1)
#             # To avoid division by zero, add epsilon
#             epsilon = 1e-10
#             sum_worker_error_rate = sum_worker_error_rate + epsilon
#             error_rates[:, i, :] = worker_error_rate / sum_worker_error_rate.unsqueeze(1)
#         return error_rates
    
#     def _e_step(self, predict_label, error_rates):
#         marginal_probability = torch.sum(predict_label, 0) / self.task_num
#         log_marginal_probability = torch.log(marginal_probability + 1e-10)
#         next_predict_label = torch.zeros([self.task_num, self.class_num], dtype=torch.float32, device=predict_label.device)

#         # Equation 2.5
#         for i in range(self.task_num):
#             log_class_likelood = self._get_log_class_likelood(error_rates, self.dataset_tensor[i])
#             log_posterior = log_marginal_probability + log_class_likelood
#             log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=0)
#             next_predict_label[i] = torch.exp(log_posterior)
#         return next_predict_label

#     def _get_likelihood(self, predict_label, error_rates):
#         log_L = 0
#         marginal_probability = torch.sum(predict_label, 0) / self.task_num
#         log_marginal_probability = torch.log(marginal_probability + 1e-10)

#         # Equation 2.7
#         for i in range(self.task_num):
#             log_class_likelood = self._get_log_class_likelood(error_rates, self.dataset_tensor[i])
#             log_posterior = log_marginal_probability + log_class_likelood
#             log_L += torch.logsumexp(log_posterior, dim=0)
#         return log_L.item()

#     def _get_log_class_likelood(self, error_rates, task_tensor):
#         # Compute the likelihood in log space
#         log_pi = torch.log(error_rates + 1e-10)  # Add epsilon to avoid log(0)
#         task_tensor_expanded = task_tensor.unsqueeze(1)  # (worker_num, 1, class_num)
#         # Multiply and sum over worker_num and observed class
#         log_class_likelood = torch.sum(task_tensor_expanded * log_pi, dim=(0,2))  # (class_num,)
#         return log_class_likelood

# faster version from chatgpt
class DawidSkeneModel:
    def __init__(self,
                 class_num,
                 max_iter=100,
                 tolerance=0.01,
                 batch_size=1000) -> None:
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size  # Adjust based on memory constraints

    def run(self, dataset):
        """
        dataset: N, H, C (supposed to be 1-hot in original impl)
        """
        self.task_num, self.worker_num, _ = dataset.shape
        self.dataset_tensor = dataset.to('cuda')  # or 'cpu' if GPU memory is insufficient
        predict_label = self.dataset_tensor.sum(1) / self.dataset_tensor.sum(1).sum(1, keepdim=True)

        flag = True
        prev_error_rates, prev_predict_label = None, None
        iter_num = 0

        while flag:
            error_rates = self._m_step(predict_label)
            next_predict_label = self._e_step(predict_label, error_rates)
            log_L = self._get_likelihood(predict_label, error_rates)

            if iter_num == 0:
                logging.info("{}\t{}".format(iter_num, log_L))
            else:
                marginal_predict = torch.sum(predict_label, 0) / self.task_num
                prev_marginal_predict = torch.sum(prev_predict_label, 0) / self.task_num
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

                print("marginals_diff", marginals_diff.item(), "error_rates_diff", error_rates_diff.item())

            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iter_num += 1

        marginal_predict = torch.sum(predict_label, 0) / self.task_num
        # Vectorized worker reliability computation
        ie_rates = marginal_predict.unsqueeze(0).unsqueeze(2) * error_rates  # [W, C, C]
        diag_ie_rates = torch.diagonal(ie_rates, dim1=1, dim2=2)  # [W, C]
        reliability = diag_ie_rates.sum(1)  # [W]
        # worker_reliability = {i: reliability[i].item() for i in range(self.worker_num)}
        
        return marginal_predict, error_rates, reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return (marginals_diff < self.tolerance and error_rates_diff < self.tolerance) or iter_num > self.max_iter

    def _m_step(self, predict_label):
        epsilon = 1e-10
        error_rates = torch.zeros((self.worker_num, self.class_num, self.class_num), dtype=torch.float32, device=predict_label.device)
        batch_size = self.batch_size  # Adjust based on memory constraints
        for start in range(0, self.worker_num, batch_size):
            end = min(start + batch_size, self.worker_num)
            batch_dataset = self.dataset_tensor[:, start:end, :]  # [T, batch_size, C]
            # Compute worker_error_rate for the batch
            worker_error_rate = torch.einsum('ti,twc->wic', predict_label, batch_dataset)
            sum_worker_error_rate = worker_error_rate.sum(2, keepdim=True) + epsilon
            error_rates[start:end] = worker_error_rate / sum_worker_error_rate
            del batch_dataset, worker_error_rate, sum_worker_error_rate
            torch.cuda.empty_cache()
        return error_rates
    
    def _e_step(self, predict_label, error_rates):
        marginal_probability = torch.sum(predict_label, 0) / self.task_num  # [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)  # [C]
        log_pi = torch.log(error_rates + 1e-10)  # [W, C, C]
        next_predict_label = torch.zeros([self.task_num, self.class_num], dtype=torch.float32, device=predict_label.device)
        batch_size = self.batch_size  # Adjust batch size based on memory constraints
        for start in range(0, self.task_num, batch_size):
            end = min(start + batch_size, self.task_num)
            batch_dataset = self.dataset_tensor[start:end]  # [batch_size, W, C]
            # Compute log_class_likelihood for the batch
            log_class_likelihood = torch.einsum('twc,wkc->tk', batch_dataset, log_pi)
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)
            next_predict_label[start:end] = torch.exp(log_posterior)
            del batch_dataset, log_class_likelihood, log_posterior
            torch.cuda.empty_cache()
        return next_predict_label

    def _get_likelihood(self, predict_label, error_rates):
        marginal_probability = torch.sum(predict_label, 0) / self.task_num  # [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)  # [C]
        log_pi = torch.log(error_rates + 1e-10)  # [W, C, C]
        log_L = 0
        batch_size = self.batch_size  # Adjust batch size based on memory constraints
        for start in range(0, self.task_num, batch_size):
            end = min(start + batch_size, self.task_num)
            batch_dataset = self.dataset_tensor[start:end]  # [batch_size, W, C]
            # Compute log_class_likelihood for the batch
            log_class_likelihood = torch.einsum('twc,wkc->tk', batch_dataset, log_pi)
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_L += torch.sum(torch.logsumexp(log_posterior, dim=1))
            del batch_dataset, log_class_likelihood, log_posterior
            torch.cuda.empty_cache()
        return log_L.item()


import torch
import logging

class DawidSkeneModelWithGold:
    def __init__(self,
                 class_num,
                 max_iter=100,
                 tolerance=0.01,
                 batch_size=1000,
                 gold_labels=None,
                 gold_weight=1.0):
        """
        Args:
            class_num (int): Number of classes.
            max_iter (int): Maximum number of EM iterations.
            tolerance (float): Convergence threshold.
            batch_size (int): Controls memory usage for large data.
            gold_labels (list or array): gold_labels[t] = c if task t has a known gold label c,
                                         or None/ -1/ some sentinel if not known.
            gold_weight (float): How much to upweight the gold-labeled tasks.
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.gold_labels = gold_labels
        self.gold_weight = gold_weight

    def run(self, dataset):
        """
        dataset: A tensor of shape (T, W, C) – T tasks, W workers, C classes.
                 Each entry dataset[t, w, c] should be 1 if worker w chose class c for task t,
                 and 0 otherwise.
        Returns:
            - marginal_predict:  (C,) the estimated overall class prior distribution.
            - error_rates:       (W, C, C) the worker confusion matrices.
            - worker_reliability: dictionary w -> scalar reliability
            - predict_label:     (T, C) the posterior distribution over labels for each task.
        """
        device = dataset.device
        self.task_num, self.worker_num, _ = dataset.shape
        self.dataset_tensor = dataset

        # =====================
        # Build task-weights:
        #   * If gold_labels[t] is known, the weight is gold_weight
        #   * Otherwise, the weight is 1.0
        # =====================
        if self.gold_labels is None:
            self.gold_labels = [None]*self.task_num
        
        # Make a torch vector of shape [T] with the appropriate weights
        self._task_weights = []
        for t in range(self.task_num):
            if self.gold_labels[t] is not None and self.gold_labels[t] != -1:
                self._task_weights.append(self.gold_weight)
            else:
                self._task_weights.append(1.0)
        self._task_weights = torch.tensor(self._task_weights, dtype=torch.float32, device=device)
        
        # =====================
        # Initialize predict_label
        #   * If gold label is known, force a one-hot vector
        #   * Otherwise, pick something like dataset.sum(1)/..., same as original
        # =====================
        naive_label = self.dataset_tensor.sum(1)  # shape (T, C)
        naive_label = naive_label / (naive_label.sum(1, keepdim=True) + 1e-10)  # normalizing row-wise
        predict_label = torch.zeros_like(naive_label)
        for t in range(self.task_num):
            if self.gold_labels[t] is not None and self.gold_labels[t] != -1:
                gold_c = self.gold_labels[t]
                # Force a delta distribution for gold-labeled tasks
                predict_label[t, :] = 0
                predict_label[t, gold_c] = 1.0
            else:
                predict_label[t, :] = naive_label[t, :]

        # EM loop
        flag = True
        prev_error_rates, prev_predict_label = None, None
        iter_num = 0

        while flag:
            # M-step
            error_rates = self._m_step(predict_label)
            # E-step
            next_predict_label = self._e_step(predict_label, error_rates)

            # Evaluate log-likelihood
            log_L = self._get_likelihood(predict_label, error_rates)

            if iter_num == 0:
                logging.info(f"{iter_num}\t{log_L}")
            else:
                # check for convergence
                marginal_predict = torch.sum(predict_label, 0) / self.task_num
                prev_marginal_predict = torch.sum(prev_predict_label, 0) / self.task_num
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

                print("Iteration:", iter_num,
                      "marginals_diff:", marginals_diff.item(),
                      "error_rates_diff:", error_rates_diff.item(),
                      "log_L:", log_L)

            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iter_num += 1

        # After final iteration
        marginal_predict = torch.sum(predict_label, 0) / self.task_num
        # Vectorized worker reliability computation
        # Weighted confusion matrix times class prior
        ie_rates = marginal_predict.unsqueeze(0).unsqueeze(2) * error_rates  # [W, C, C]
        diag_ie_rates = torch.diagonal(ie_rates, dim1=1, dim2=2)            # [W, C]
        reliability = diag_ie_rates.sum(1)                                  # [W]
        worker_reliability = {i: reliability[i].item() for i in range(self.worker_num)}

        return marginal_predict, error_rates, worker_reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return ((marginals_diff < self.tolerance and error_rates_diff < self.tolerance)
                or iter_num > self.max_iter)

    def _m_step(self, predict_label):
        """
        Incorporate task-weights in computing worker confusion matrices.
        """
        epsilon = 1e-10
        error_rates = torch.zeros(
            (self.worker_num, self.class_num, self.class_num),
            dtype=torch.float32,
            device=predict_label.device
        )

        batch_size = self.batch_size
        # For convenience: shape [T, 1], so we can do broadcast in einsum
        task_weights_ = self._task_weights.view(-1, 1)

        for start in range(0, self.worker_num, batch_size):
            end = min(start + batch_size, self.worker_num)
            batch_dataset = self.dataset_tensor[:, start:end, :]   # [T, batch_size, C]
            # Weighted version:
            #   worker_error_rate[w, i, j] ~ sum_{t} [ w_t * p(i|t) * 1(worker w labeled j) ]
            # We'll do: (T, i) x (T, w, j)
            #   with an additional factor of w_t
            # So shape manipulations matter. One approach:
            #   worker_error_rate = torch.einsum('t,ti,twj->wi j', w, predict_label, dataset)
            # but we also must handle the batch dimension for w, so we do w' in [w_start, w_end].
            # We can still do the same approach if we dimension-match carefully:
            worker_error_rate = torch.einsum(
                't,ti,twc->wic',
                task_weights_.squeeze(-1),
                predict_label,
                batch_dataset
            )
            sum_worker_error_rate = worker_error_rate.sum(2, keepdim=True) + epsilon
            error_rates[start:end] = worker_error_rate / sum_worker_error_rate

            # Clean up
            del batch_dataset, worker_error_rate, sum_worker_error_rate
            torch.cuda.empty_cache()

        return error_rates

    def _e_step(self, predict_label, error_rates):
        """
        E-step with partial override for gold tasks:
          If task t has a gold label c, then next_predict_label[t] = one-hot(c)
          Otherwise, next_predict_label[t] is obtained in the usual DS manner.
        """
        device = predict_label.device
        marginal_probability = torch.sum(predict_label, 0) / self.task_num  # [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)  # [C]
        log_pi = torch.log(error_rates + 1e-10)                             # [W, C, C]

        next_predict_label = torch.zeros_like(predict_label)
        batch_size = self.batch_size

        # We'll fill next_predict_label in chunks.
        for start in range(0, self.task_num, batch_size):
            end = min(start + batch_size, self.task_num)
            batch_dataset = self.dataset_tensor[start:end]  # [batch_size, W, C]

            # For tasks that are NOT gold-labeled, do the usual DS update
            # For tasks that ARE gold-labeled, skip directly to a one-hot

            # 1) compute log_class_likelihood for these tasks
            log_class_likelihood = torch.einsum('twc,wkc->tk', batch_dataset, log_pi)
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood

            # 2) for each task in [start, end], if it's not gold-labeled, do normal normalization
            #    otherwise, set distribution = one-hot.
            for local_t in range(start, end):
                t_idx = local_t - start  # index in this batch
                gold_c = self.gold_labels[local_t]
                if gold_c is not None and gold_c != -1:
                    # Force the distribution to be a delta at gold_c
                    next_predict_label[local_t, :] = 0
                    next_predict_label[local_t, gold_c] = 1.0
                else:
                    # normal DS
                    row = log_posterior[t_idx]
                    row = row - torch.logsumexp(row, dim=0, keepdim=True)
                    next_predict_label[local_t] = torch.exp(row)

            del batch_dataset, log_class_likelihood, log_posterior
            torch.cuda.empty_cache()

        return next_predict_label

    def _get_likelihood(self, predict_label, error_rates):
        """
        Incorporate task-weights in the log-likelihood.
        """
        marginal_probability = torch.sum(predict_label, 0) / self.task_num  # [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)  # [C]
        log_pi = torch.log(error_rates + 1e-10)  # [W, C, C]

        log_L = 0.0
        batch_size = self.batch_size

        # We'll need to multiply by self._task_weights to upweight gold tasks
        for start in range(0, self.task_num, batch_size):
            end = min(start + batch_size, self.task_num)
            batch_dataset = self.dataset_tensor[start:end]  # [batch_size, W, C]

            # shape = (batch_size,)
            w_batch = self._task_weights[start:end]

            # log_class_likelihood: shape [batch_size, C]
            log_class_likelihood = torch.einsum('twc,wkc->tk', batch_dataset, log_pi)
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood

            # logsumexp over classes -> shape [batch_size]
            # Then multiply by the weights and sum
            # log_L += \sum_{t} w_t * log( sum_c p(c) * L_t(c) )
            per_task_term = torch.logsumexp(log_posterior, dim=1)  # (batch_size,)
            log_L += torch.sum(w_batch * per_task_term)

            del batch_dataset, w_batch, log_class_likelihood, log_posterior, per_task_term
            torch.cuda.empty_cache()

        return log_L.item()




import torch
import torch.nn.functional as F
import logging

class DawidSkeneModelWithCalibration:
    """Observed results: same as normal DS; still uses confusion matrices"""
    def __init__(
        self,
        class_num,
        max_iter=100,
        tolerance=1e-3,
        batch_size=1000,
        lr_temp=1e-2
    ):
        """
        EM-based Dawid-Skene for multi-class labeling with:
         - Logits from each 'worker' or classifier
         - Per-worker temperature calibration
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.lr_temp = lr_temp

        # Will be set at runtime:
        self.task_num = None
        self.worker_num = None
        self.dataset_tensor = None  # shape [T, W, C]
        self.temperatures = None    # shape [W], requires_grad = True
        self.temp_optimizer = None

    def run(self, dataset):
        """
        dataset: shape [T, W, C], with logits from each worker on each item.
        Returns:
          marginal_predict, error_rates, worker_reliability, predict_label
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_tensor = dataset.to(device)
        self.task_num, self.worker_num, _ = self.dataset_tensor.shape

        # 1) Initialize per-worker temperature at 1.0
        self.temperatures = torch.ones(self.worker_num, device=device, requires_grad=True)
        self.temp_optimizer = torch.optim.Adam([self.temperatures], lr=self.lr_temp)

        # 2) Initialize predict_label => average softmax at T=1
        with torch.no_grad():
            worker_probs = F.softmax(self.dataset_tensor, dim=2)   # [T, W, C]
            predict_label = torch.mean(worker_probs, dim=1)        # [T, C]
        predict_label = predict_label.detach()

        prev_error_rates = None
        prev_predict_label = None

        iteration = 0
        still_running = True
        while still_running:
            # --- M-step (updates confusion + temperatures) ---
            error_rates = self._m_step(predict_label)  # confusion => no grad; temperature => grad
            error_rates = error_rates.detach()          # ensure no leftover graph

            # --- E-step (purely no-grad) ---
            with torch.no_grad():
                next_predict_label = self._e_step(predict_label, error_rates)
            next_predict_label = next_predict_label.detach()

            # Evaluate log-likelihood for monitoring (no grad)
            with torch.no_grad():
                log_L = self._get_likelihood(next_predict_label, error_rates)

            # Check diffs for stopping
            if iteration == 0:
                logging.info(f"Iter={iteration}, log_L={log_L:.4f}")
            else:
                marginal_predict = torch.sum(next_predict_label, dim=0) / self.task_num
                prev_marginal_predict = torch.sum(prev_predict_label, dim=0) / self.task_num
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                logging.info(
                    f"Iter={iteration}, log_L={log_L:.4f}, "
                    f"marginals_diff={marginals_diff:.6f}, error_rates_diff={error_rates_diff:.6f}"
                )
                if self._check_condition(marginals_diff, error_rates_diff, iteration):
                    still_running = False

                print("marginals_diff", marginals_diff.item(), "error_rates_diff", error_rates_diff.item())

            # Update for next iteration
            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iteration += 1

        # Final marginals
        marginal_predict = torch.sum(predict_label, dim=0) / self.task_num

        # Worker reliability = sum of diag of error_rates * class prevalence
        ie_rates = marginal_predict.unsqueeze(0).unsqueeze(2) * error_rates  # [W, C, C]
        diag_ie_rates = torch.diagonal(ie_rates, dim1=1, dim2=2)            # [W, C]
        reliability = diag_ie_rates.sum(dim=1)                              # [W]
        worker_reliability = {int(w): reliability[w].item() for w in range(self.worker_num)}

        return marginal_predict, error_rates, worker_reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        cond1 = (marginals_diff < self.tolerance and error_rates_diff < self.tolerance)
        cond2 = (iter_num > self.max_iter)
        return cond1 or cond2

    def _m_step(self, predict_label):
        """
        M-step:
          - Recompute confusion matrices (no grad)
          - Then do sub-steps to optimize self.temperatures by gradient
        """
        # (1) confusion matrices
        with torch.no_grad():
            error_rates = self._recompute_confusion(predict_label)
        error_rates = error_rates.detach()

        # (2) temperature updates with gradient
        self._update_temperatures(predict_label, error_rates, steps=5)

        return error_rates

    @torch.no_grad()
    def _recompute_confusion(self, predict_label):
        """
        Recompute confusion matrices. No grad needed because these are just counts.
        """
        epsilon = 1e-10
        device = predict_label.device

        error_rates = torch.zeros((self.worker_num, self.class_num, self.class_num),
                                  dtype=torch.float32, device=device)

        for start in range(0, self.worker_num, self.batch_size):
            end = min(start + self.batch_size, self.worker_num)

            # [T, Wb, C]
            batch_logits = self.dataset_tensor[:, start:end, :]
            # shape [Wb]
            batch_temps = self.temperatures[start:end]

            scaled = torch.einsum('twc,w->twc', batch_logits, 1.0 / batch_temps)
            batch_probs = F.softmax(scaled, dim=2)   # [T, Wb, C]

            # DS formula: sum_i [p_i(c)*batch_probs(i,w,k)]
            worker_error_rate = torch.einsum(
                'tc,twk->wck',
                predict_label,    # [T, C]
                batch_probs       # [T, Wb, C]
            )
            sums = worker_error_rate.sum(dim=2, keepdim=True) + epsilon
            error_rates[start:end] = worker_error_rate / sums

            del batch_logits, batch_temps, scaled, batch_probs, worker_error_rate, sums
            torch.cuda.empty_cache()

        return error_rates

    def _update_temperatures(self, predict_label, error_rates, steps=5):
        """
        Multiple small gradient steps on self.temperatures
        Each step = new forward pass => calc neg_log_lik => backward => step
        """
        for _ in range(steps):
            self.temp_optimizer.zero_grad()
            neg_log_lik = -self._calc_log_likelihood_for_temp(predict_label, error_rates)
            neg_log_lik.backward()    # Freed at this line
            self.temp_optimizer.step()

            with torch.no_grad():
                self.temperatures.clamp_(min=1e-5)

    def _calc_log_likelihood_for_temp(self, predict_label, error_rates):
        """
        Forward pass that depends on self.temperatures, returns scalar Tensor -> backward.
        No .item() calls here; keep it as a connected graph.
        """
        device = predict_label.device

        # Class prior
        marginal_probability = torch.sum(predict_label, dim=0) / self.task_num  # [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)      # [C]

        log_lik = torch.zeros([], device=device, dtype=torch.float32)

        for start in range(0, self.task_num, self.batch_size):
            end = min(start + self.batch_size, self.task_num)
            batch_logits = self.dataset_tensor[start:end]   # [B, W, C]

            # scale by temperature
            scaled = torch.einsum('bwc,w->bwc', batch_logits, 1.0 / self.temperatures)
            batch_probs = F.softmax(scaled, dim=2)          # [B, W, C]

            partial = torch.einsum('bwc,wck->bwc', batch_probs, error_rates)
            logpartial = torch.log(partial + 1e-10).sum(dim=1)  # [B, C]

            log_posterior = log_marginal_probability.unsqueeze(0) + logpartial
            # sum over c => logsumexp => shape: [B]
            log_lik_batch = torch.logsumexp(log_posterior, dim=1).sum()

            log_lik = log_lik + log_lik_batch

            del batch_logits, scaled, batch_probs, partial, logpartial, log_posterior
            torch.cuda.empty_cache()

        return log_lik

    @torch.no_grad()
    def _e_step(self, old_predict_label, error_rates):
        """
        E-step => next_predict_label[i,c] ∝ pi[c] * ∏_w sum_k [p_worker(i,w,k)*error_rates[w,c,k]]
        No gradient needed => no_grad
        """
        device = old_predict_label.device
        batch_size = self.batch_size

        # class prior
        marginal_probability = torch.sum(old_predict_label, dim=0) / self.task_num
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        next_predict_label = torch.zeros([self.task_num, self.class_num],
                                         dtype=torch.float32, device=device)

        for start in range(0, self.task_num, batch_size):
            end = min(start + batch_size, self.task_num)

            batch_logits = self.dataset_tensor[start:end]   # [B, W, C]
            scaled = torch.einsum('bwc,w->bwc', batch_logits, 1.0 / self.temperatures)
            batch_probs = F.softmax(scaled, dim=2)          # [B, W, C]

            partial = torch.einsum('bwc,wck->bwc', batch_probs, error_rates)
            logpartial = torch.log(partial + 1e-10).sum(dim=1)  # [B, C]

            log_posterior = log_marginal_probability.unsqueeze(0) + logpartial
            # normalize across c
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)

            next_predict_label[start:end] = torch.exp(log_posterior)

            del batch_logits, scaled, batch_probs, partial, logpartial, log_posterior
            torch.cuda.empty_cache()

        return next_predict_label

    @torch.no_grad()
    def _get_likelihood(self, predict_label, error_rates):
        """
        Logging only. Return float log-likelihood. No grad => no effect on temperature graph.
        """
        device = predict_label.device
        batch_size = self.batch_size

        marginal_probability = torch.sum(predict_label, dim=0) / self.task_num
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        log_L_total = 0.0
        for start in range(0, self.task_num, batch_size):
            end = min(start + batch_size, self.task_num)

            batch_logits = self.dataset_tensor[start:end]
            scaled = torch.einsum('bwc,w->bwc', batch_logits, 1.0 / self.temperatures)
            batch_probs = F.softmax(scaled, dim=2)

            partial = torch.einsum('bwc,wck->bwc', batch_probs, error_rates)
            logpartial = torch.log(partial + 1e-10).sum(dim=1)
            log_posterior = log_marginal_probability.unsqueeze(0) + logpartial

            log_L_batch = torch.logsumexp(log_posterior, dim=1).sum().item()
            log_L_total += log_L_batch

            del batch_logits, scaled, batch_probs, partial, logpartial, log_posterior
            torch.cuda.empty_cache()

        return log_L_total

# import torch
# import torch.nn.functional as F
# import logging

# def partial_log_product(batch_probs, group_size=20):
#     """
#     batch_probs: (B, W, C) probabilities
#     We'll sum logs in groups of 'group_size' workers to mitigate huge negative sums.
#     """
#     B, W, C = batch_probs.shape
#     device = batch_probs.device

#     # Start the partial log product with the first chunk
#     cur_logprod = stable_log_sum_of_logs(torch.log(batch_probs[:, :group_size, :].clamp_min(1e-40)), dim=1)
#     # 'cur_logprod' shape => [B, C], representing sum_of_logs for the first group_size workers

#     idx = group_size
#     while idx < W:
#         next_idx = min(idx + group_size, W)
#         chunk_log = torch.log(batch_probs[:, idx:next_idx, :].clamp_min(1e-40))
#         chunk_sum = stable_log_sum_of_logs(chunk_log, dim=1)  # [B, C]
#         # Add to the existing sum of logs
#         cur_logprod = cur_logprod + chunk_sum
#         idx = next_idx

#     return cur_logprod

# def stable_log_sum_of_logs(logvals, dim):
#     """
#     Sums logvals along 'dim' in a stable manner by subtracting the max.
#     This is effectively sum_w logvals[w], but carefully.
#     """
#     # logvals shape: e.g. (B, group_size, C)
#     m = logvals.max(dim=dim, keepdim=True).values
#     shifted = logvals - m
#     exp_shifted = shifted.exp().sum(dim=dim)
#     return m.squeeze(dim) + torch.log(exp_shifted.clamp_min(1e-40))


# class DawidSkeneNoConfusion:
#     """
#     An EM-like aggregator that:
#       - Has a global class prior p(c).
#       - For each worker w, learns a temperature T_w that scales the worker's logits.
#       - For each item i, computes a posterior p_i(c).
#     No confusion matrices are used.
#     """
#     def __init__(
#         self,
#         class_num,
#         max_iter=50,
#         tolerance=1e-6,
#         batch_size=1000,
#         lr_temp=1e-4
#     ):
#         """
#         Args:
#             class_num (int): Number of classes, C.
#             max_iter (int): Maximum number of EM iterations.
#             tolerance (float): Convergence threshold for changes in p(c) or p_i(c).
#             batch_size (int): For chunking large data when computing log-likelihood.
#             lr_temp (float): Learning rate for temperature updates (Adam).
#         """
#         self.class_num = class_num
#         self.max_iter = max_iter
#         self.tolerance = tolerance
#         self.batch_size = batch_size
#         self.lr_temp = lr_temp

#         # Will be set at runtime:
#         self.task_num = None    # number of items T
#         self.worker_num = None  # number of workers W
#         self.dataset_tensor = None  # shape [T, W, C] of logits
#         self.temperatures = None    # shape [W], learnable
#         self.temp_optimizer = None

#         # class prior p(c)
#         self.class_prior = None       # shape [C], not learnable by gradient, but updated in M-step

#     def run(self, dataset):
#         """
#         Main entry point.  Expects 'dataset' with shape [T, W, C] = (logits).
#         Returns:
#           final_class_prior: shape [C], p(c)
#           final_temps: shape [W], the learned temperatures
#           final_posteriors: shape [T, C], p_i(c) for each item
#         """
#         # clamp logits to avoid underflow
#         device = dataset.device
#         self.dataset_tensor = torch.clamp(dataset, min=-20.0, max=20.0)
#         self.task_num, self.worker_num, _ = self.dataset_tensor.shape

#         # 1) Initialize p(c) to uniform
#         self.class_prior = torch.ones(self.class_num, device=device)
#         self.class_prior /= self.class_prior.sum()

#         # 2) Initialize per-worker temperature at 1.0
#         self.temperatures = torch.ones(self.worker_num, device=device, requires_grad=True)
#         self.temp_optimizer = torch.optim.Adam([self.temperatures], lr=self.lr_temp)

#         # 3) Initialize item posteriors p_i(c) by a naive approach:
#         #    For each item, average the worker's T=1 softmax probabilities and normalize
#         with torch.no_grad():
#             worker_probs = F.softmax(self.dataset_tensor, dim=2)      # [T, W, C]
#             p_i = torch.mean(worker_probs, dim=1)                     # [T, C]
#             p_i = p_i / p_i.sum(dim=1, keepdim=True).clamp_min(1e-10) # normalize
#         posteriors = p_i.clone()

#         iteration = 0
#         converged = False

#         # We'll track the old posteriors for a convergence check
#         old_posteriors = None

#         while not converged and iteration < self.max_iter:
#             # ================== E-step ==================
#             with torch.no_grad():
#                 # Update item posteriors p_i(c)
#                 new_posteriors = self._e_step(posteriors)

#             # ================== M-step ==================
#             # M-step part 1: update class_prior by closed form
#             with torch.no_grad():
#                 alpha = 1e-3
#                 self.class_prior = new_posteriors.mean(dim=0) + alpha  # average across items => shape [C]
#                 self.class_prior /= self.class_prior.sum()             # normalize

#             # M-step part 2: update temperatures by gradient
#             self._update_temperatures(new_posteriors)

#             # Evaluate changes
#             diff = torch.tensor(0.0)
#             if old_posteriors is not None:
#                 diff = torch.sum(torch.abs(new_posteriors - old_posteriors))
#                 if diff.item() < self.tolerance:
#                     converged = True

#                 # print("posteriors diff", diff.item())


#             old_posteriors = new_posteriors
#             posteriors = new_posteriors
#             iteration += 1

#             # (Optional) Logging
#             logL = self._calc_log_likelihood(posteriors).item()
#             # logging.info(f"Iter={iteration}, logL={logL:.4f}, diff={diff.item() if old_posteriors is not None else 0.0}")
#             print(f"Iter={iteration}, logL={logL:.4f}, diff={diff.item() if old_posteriors is not None else 0.0}")

#         # Final results
#         final_class_prior = self.class_prior.clone().detach()
#         final_temps = self.temperatures.detach()
#         final_posteriors = posteriors.detach()

#         return final_class_prior, final_temps, final_posteriors

#     @torch.no_grad()
#     def _e_step(self, old_posteriors):
#         """
#         E-step: For each item i, we do
#           new_p_i(c) ∝ p(c) * ∏_w softmax(logits[i,w,:]/temp[w])[c]
#         Then normalize over c.

#         Args:
#             old_posteriors: shape [T, C], not used here except for logging if needed
#         Returns:
#             new_posteriors: shape [T, C]
#         """
#         T, W, C = self.dataset_tensor.shape
#         device = self.dataset_tensor.device

#         # We'll build new_p_i in a chunked manner to handle big T
#         new_posteriors = torch.zeros((T, C), device=device)

#         for start in range(0, T, self.batch_size):
#             end = min(start + self.batch_size, T)
#             # shape [batch_size, W, C]
#             batch_logits = self.dataset_tensor[start:end]

#             # shape [W] => broadcast
#             # print("self.temperatures", self.temperatures)
#             scaled = torch.einsum('bwc,w->bwc', batch_logits, 1.0 / self.temperatures)
#             # shape [batch_size, W, C]
#             batch_probs = F.softmax(scaled, dim=2)
#             # print("batch_probs", batch_probs)
#             # print('batch probs[0][0]', batch_probs[0][0].min(), batch_probs[0][0].max())

#             # 1.
#             # compute product over w of batch_probs[i,w,c]
#             # shape => [batch_size, C]
#             # We'll do exp( sum_w log(...) ) approach to avoid underflow
#             # or we can just multiply directly if T is not huge:

#             # # log_probs[i,c] = sum_w log( batch_probs[i,w,c] )
#             # log_probs = torch.log(batch_probs.clamp_min(1e-10)).sum(dim=1)  # [batch_size, C]
#             # print("log_probs for item i:", log_probs)

#             # # then exponentiate
#             # product_probs = torch.exp(log_probs)  # shape [batch_size, C]
#             # 
#             # product_probs = product_probs.clamp_min(1e-30)
            
#             # 2.
#             # # We want log_product[i,c] = sum_w log( p_{i,w}(c) ).
#             # # But do it in a stable way:
#             # log_p = torch.log(batch_probs.clamp_min(1e-40))  # shape [B, W, C]
#             # print("log p ", log_p)

#             # # 1) find the maximum across w for each (i,c) 
#             # max_per_ic = log_p.max(dim=1, keepdim=True).values  # shape [B, 1, C]
#             # print('max_per_ic',max_per_ic)

#             # # 2) subtract it
#             # shifted = log_p - max_per_ic
#             # print('shifted', shifted)

#             # # 3) now sum
#             # sum_shifted = shifted.sum(dim=1)  # shape [B, C]
#             # print('sum_shifted', sum_shifted)

#             # # 4) exponentiate + re-shift
#             # product_probs = torch.exp(sum_shifted) * torch.exp(max_per_ic.sum(dim=1))
#             # print("product_probs for item i:", product_probs)

#             # 3.
#             log_probs = partial_log_product(batch_probs, group_size=20)  # shape [B, C]
#             print("log_probs for item i:", log_probs)
#             product_probs = torch.exp(log_probs)
#             print("product_probs for item i:", product_probs)

#             # multiply by class_prior
#             # shape [C]
#             product_probs = product_probs * self.class_prior.unsqueeze(0)
#             print('times class prior', product_probs)

#             # normalize over c
#             product_probs /= product_probs.sum(dim=1, keepdim=True).clamp_min(1e-10)
#             print('normalized over c', product_probs)

#             new_posteriors[start:end] = product_probs

#         return new_posteriors

#     def _update_temperatures(self, posteriors, steps=5):
#         """
#         M-step subroutine: multiple gradient steps on the negative log-likelihood wrt. self.temperatures
#         """
#         for _ in range(steps):
#             self.temp_optimizer.zero_grad()
#             neg_log_lik = -self._calc_log_likelihood(posteriors)  # forward pass
#             neg_log_lik.backward()       # single backward => frees graph
#             self.temp_optimizer.step()

#             # clamp temps to avoid zero or negative
#             with torch.no_grad():
#                 # self.temperatures.clamp_(min=1e-4)
#                 self.temperatures.clamp_(min=1e-3, max=50.0)

#     def _calc_log_likelihood(self, posteriors):
#         """
#         The log-likelihood:
#           L = sum_i log( sum_c [ p(c)* ∏_w softmax(logits[i,w]/temp[w])[c] ] )
#         We won't do an internal E-step here, because we already have p(c).
#         We'll chunk the sum over items if T is large.
#         Returns a scalar tensor.
#         """
#         device = self.dataset_tensor.device
#         T, W, C = self.dataset_tensor.shape
#         batch_size = self.batch_size

#         # We'll accumulate as a single scalar
#         logL = torch.zeros([], device=device)

#         # We'll do: sum_i log( sum_c [ class_prior[c] * ∏_w pi_{i,w}(c) ] )
#         for start in range(0, T, batch_size):
#             end = min(start + batch_size, T)
#             batch_logits = self.dataset_tensor[start:end]  # shape [B, W, C]
#             # shape [B, W, C]
#             scaled = torch.einsum('bwc,w->bwc', batch_logits, 1.0 / self.temperatures)
#             batch_probs = F.softmax(scaled, dim=2)  # shape [B, W, C]

#             # product over w => we'll do log-sum-exp with the known prior
#             # log_probs[i,c] = sum_w log( batch_probs[i,w,c] ) + log( class_prior[c] )
#             log_probs = torch.log(batch_probs.clamp_min(1e-10)).sum(dim=1)  # [B, C]
#             log_probs = log_probs + torch.log(self.class_prior + 1e-10).unsqueeze(0)  # broadcast

#             # sum_c => logsumexp
#             # shape [B]
#             batch_logsum = torch.logsumexp(log_probs, dim=1)
#             logL += batch_logsum.sum()

#         return logL


import torch
import torch.nn.functional as F

def partial_sum_of_logs(batch_probs, group_size=20):
    """
    Correct function to compute sum_{w in groups} log( prob[i,w,c] ) 
    in small chunks, avoiding extremely large negative or positive sums.

    batch_probs: (B, W, C) = probabilities for B items, W workers, C classes.
    We return a tensor of shape [B, C] with the sum of log(prob) over all W workers.

    1) We do chunked summation to keep partial sums smaller,
       which helps avoid extreme underflow/overflow.
    2) We clamp the final sums to e.g. [-700, 700] to keep exponent in a safe range.
    """
    B, W, C = batch_probs.shape
    device = batch_probs.device

    # We'll store the running sum of logs in out_log. Start at 0.
    out_log = torch.zeros((B, C), device=device)

    idx = 0
    while idx < W:
        end = min(idx + group_size, W)
        # shape = [B, chunk_size, C]
        chunk = batch_probs[:, idx:end, :].clamp_min(1e-40)
        # sum of log(prob) over this chunk's workers
        sum_log_chunk = torch.log(chunk).sum(dim=1)  # => [B, C]
        # add it to out_log
        out_log += sum_log_chunk
        idx = end

    # Optional clamp to avoid huge exponent
    out_log = out_log.clamp(min=-700.0, max=700.0)
    return out_log


class DawidSkeneNoConfusion:
    """
    An EM-like aggregator that:
      - Has a global class prior p(c).
      - For each worker w, learns a temperature T_w that scales the worker's logits.
      - For each item i, computes a posterior p_i(c).
    No confusion matrices are used; we directly form products of each worker's softmax.
    """
    def __init__(
        self,
        class_num,
        max_iter=50,
        tolerance=1e-6,
        batch_size=1000,
        lr_temp=1e-4
    ):
        """
        Args:
            class_num (int): Number of classes, C.
            max_iter (int): Maximum number of EM iterations.
            tolerance (float): Convergence threshold for changes in p(c) or p_i(c).
            batch_size (int): For chunking large data when computing log-likelihood.
            lr_temp (float): Learning rate for temperature updates (Adam).
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.lr_temp = lr_temp

        self.task_num = None    
        self.worker_num = None  
        self.dataset_tensor = None  
        self.temperatures = None    
        self.temp_optimizer = None

        # class prior p(c)
        self.class_prior = None

    def run(self, dataset):
        """
        Main entry point.  Expects 'dataset' with shape [T, W, C] = logits.
        Returns:
          final_class_prior: shape [C], p(c)
          final_temps: shape [W], the learned temperatures
          final_posteriors: shape [T, C], p_i(c) for each item
        """
        device = dataset.device
        # 1) Clamp raw logits to [-20, 20] to avoid extremely large exponents
        self.dataset_tensor = torch.clamp(dataset, min=-20.0, max=20.0)
        self.task_num, self.worker_num, _ = self.dataset_tensor.shape

        # 2) Initialize p(c) to uniform
        self.class_prior = torch.ones(self.class_num, device=device)
        self.class_prior /= self.class_prior.sum()

        # 3) Initialize per-worker temperature at 1.0
        self.temperatures = torch.ones(self.worker_num, device=device, requires_grad=True)
        self.temp_optimizer = torch.optim.Adam([self.temperatures], lr=self.lr_temp)

        # 4) Initialize item posteriors p_i(c) by naive approach:
        with torch.no_grad():
            worker_probs = F.softmax(self.dataset_tensor, dim=2)  # [T, W, C]
            p_i = torch.mean(worker_probs, dim=1)                 # [T, C]
            p_i = p_i / p_i.sum(dim=1, keepdim=True).clamp_min(1e-10)
        posteriors = p_i.clone()

        iteration = 0
        converged = False
        old_posteriors = None

        while not converged and iteration < self.max_iter:
            # ============= E-step =============
            with torch.no_grad():
                new_posteriors = self._e_step()

            # ============= M-step =============
            # part 1: update class_prior (closed form)
            with torch.no_grad():
                alpha = 1e-3
                self.class_prior = new_posteriors.mean(dim=0) + alpha
                self.class_prior /= self.class_prior.sum()

            # part 2: update temperatures by gradient
            self._update_temperatures(new_posteriors, steps=5)

            # check difference in posteriors
            diff = torch.tensor(0.0, device=device)
            if old_posteriors is not None:
                diff = torch.sum(torch.abs(new_posteriors - old_posteriors))
                if diff.item() < self.tolerance:
                    converged = True

            old_posteriors = new_posteriors
            posteriors = new_posteriors
            iteration += 1

            # log-likelihood for debugging
            logL = self._calc_log_likelihood(posteriors).item()
            print(f"Iter={iteration}, logL={logL:.4f}, diff={diff.item():.4g}")

        # Final results
        final_class_prior = self.class_prior.clone().detach()
        final_temps = self.temperatures.detach()
        final_posteriors = posteriors.detach()

        return final_class_prior, final_temps, final_posteriors

    @torch.no_grad()
    def _e_step(self):
        """
        E-step: For each item i:
          p_i(c) ∝ p(c) * ∏_w softmax(logits[i,w,:]/temp[w])[c]
        Then we normalize. We do chunked processing over items.
        """
        T, W, C = self.dataset_tensor.shape
        device = self.dataset_tensor.device

        new_posteriors = torch.zeros((T, C), device=device)

        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            batch_logits = self.dataset_tensor[start:end]     # shape [B, W, C]
            scaled = torch.einsum('bwc,w->bwc', batch_logits, 1.0 / self.temperatures)
            batch_probs = F.softmax(scaled, dim=2)            # [B, W, C]

            # sum_of_logs[i,c] = Σ_{w} log( batch_probs[i,w,c] )
            sum_of_logs = partial_sum_of_logs(batch_probs, group_size=20)  # [B, C]

            # exponentiate => product of prob
            product_probs = torch.exp(sum_of_logs)  # shape [B, C]

            # multiply by class prior
            product_probs *= self.class_prior.unsqueeze(0)

            # normalize over c
            product_probs /= product_probs.sum(dim=1, keepdim=True).clamp_min(1e-40)

            new_posteriors[start:end] = product_probs

        return new_posteriors

    def _update_temperatures(self, posteriors, steps=5):
        """
        M-step subroutine: multiple gradient steps on negative log-likelihood wrt. self.temperatures
        """
        for _ in range(steps):
            self.temp_optimizer.zero_grad()
            neg_log_lik = -self._calc_log_likelihood(posteriors)
            neg_log_lik.backward()
            self.temp_optimizer.step()

            with torch.no_grad():
                # clamp temperatures to a reasonable range
                self.temperatures.clamp_(min=1e-3, max=50.0)

    def _calc_log_likelihood(self, posteriors):
        """
        The total log-likelihood: sum_i log( sum_c [ p(c)* ∏_w pi_{i,w}(c) ] ).
        We'll chunk over items. We'll do a direct sum of logs approach. 
        """
        T, W, C = self.dataset_tensor.shape
        device = self.dataset_tensor.device
        logL = torch.zeros([], device=device)

        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            batch_logits = self.dataset_tensor[start:end]  # [B, W, C]
            scaled = torch.einsum('bwc,w->bwc', batch_logits, 1.0 / self.temperatures)
            batch_probs = F.softmax(scaled, dim=2)         # [B, W, C]

            # sum_of_logs[i,c] = Σ_{w} log(batch_probs[i,w,c])
            sum_of_logs = torch.log(batch_probs.clamp_min(1e-40)).sum(dim=1)  # [B, C]
            # add log class prior
            sum_of_logs += torch.log(self.class_prior + 1e-40).unsqueeze(0)

            # log-sum-exp over c
            batch_logsum = torch.logsumexp(sum_of_logs, dim=1)
            logL += batch_logsum.sum()

        return logL


#
# That’s it!
#
# Key fixes:
#  1) Use partial_sum_of_logs to do a direct sum of logs (not log-sum-exp).
#  2) Clamp sums to avoid exp() => inf.
#  3) Properly clamp temperatures & logits.
#  4) No "stable_log_sum_of_logs" that confused summation of logs with log-sum-exp.
#



import torch
import torch.nn.functional as F

class LargeWorkerAggregator:
    def __init__(
        self,
        class_num: int,
        max_iter=50,
        tolerance=1e-6,
        batch_size=1000,
        lr_temp=1e-4,
        worker_chunk=50,
        item_batch=100
    ):
        """
        :param class_num: number of classes
        :param max_iter: max EM iterations
        :param tolerance: threshold for stopping
        :param batch_size: item chunk size for the E-step
        :param lr_temp: learning rate for temperature updates
        :param worker_chunk: how many workers to process at once in partial-dist
        :param item_batch: how many items per micro-batch for gradient accumulation
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.lr_temp = lr_temp
        self.worker_chunk = worker_chunk
        self.item_batch = item_batch

        self.task_num = None
        self.worker_num = None
        self.dataset_tensor = None
        self.temperatures = None
        self.temp_optimizer = None

        self.class_prior = None

    def run(self, dataset: torch.Tensor):
        device = dataset.device
        self.dataset_tensor = torch.clamp(dataset, min=-20.0, max=20.0)
        T, W, C = self.dataset_tensor.shape
        self.task_num = T
        self.worker_num = W

        # Initialize prior
        self.class_prior = torch.ones(C, device=device)
        self.class_prior /= self.class_prior.sum()

        # Initialize temps
        self.temperatures = torch.ones(W, device=device, requires_grad=True)
        self.temp_optimizer = torch.optim.Adam([self.temperatures], lr=self.lr_temp)

        # Initialize posteriors from T=1
        with torch.no_grad():
            worker_probs = F.softmax(self.dataset_tensor, dim=2)  # [T, W, C]
            p_i = torch.mean(worker_probs, dim=1)                 # [T, C]
            p_i /= p_i.sum(dim=1, keepdim=True).clamp_min(1e-10)
        posteriors = p_i.clone()

        iteration = 0
        old_posteriors = None
        converged = False

        while not converged and iteration < self.max_iter:
            # E-step
            posteriors = self._e_step()

            # M-step part 1: update prior
            with torch.no_grad():
                alpha = 1e-3
                new_prior = posteriors.mean(dim=0) + alpha
                new_prior /= new_prior.sum()
                self.class_prior = new_prior

            # M-step part 2: update temps
            self._update_temperatures(posteriors, steps=5)

            # Check diff
            diff = torch.tensor(0.0, device=device)
            if old_posteriors is not None:
                diff = (posteriors.detach() - old_posteriors).abs().sum()
                if diff.item() < self.tolerance:
                    converged = True

            old_posteriors = posteriors.detach()
            iteration += 1

            # log-likelihood for logging
            ll = self._calc_log_likelihood_no_grad()
            print(f"Iter={iteration}, logL={ll:.4f}, diff={diff.item():.4g}")

        return self.class_prior.detach().clone(), self.temperatures.detach().clone(), posteriors.detach().clone()

    def _e_step(self):
        # Same logic as before, but no "no_grad" so it can build some graph
        # Actually, for a classical EM approach, you might not need a graph here.
        # We'll keep it out of no_grad so there's no confusion.
        T, W, C = self.dataset_tensor.shape
        device = self.dataset_tensor.device
        new_posteriors = []

        for start_item in range(0, T, self.batch_size):
            end_item = min(start_item + self.batch_size, T)
            B = end_item - start_item

            partial_dist = torch.ones((B, C), device=device)

            widx = 0
            while widx < W:
                wend = min(widx + self.worker_chunk, W)
                chunk_logits = self.dataset_tensor[start_item:end_item, widx:wend, :]
                chunk_scaled = torch.einsum('bwc,w->bwc', chunk_logits, 1.0 / self.temperatures[widx:wend])
                chunk_probs = F.softmax(chunk_scaled, dim=2).clamp_min(1e-40)

                partial_dist = self._progressive_multiply(partial_dist, chunk_probs)
                widx = wend

            partial_dist *= self.class_prior.unsqueeze(0)
            sums = partial_dist.sum(dim=1, keepdim=True).clamp_min(1e-40)
            partial_dist /= sums

            new_posteriors.append(partial_dist)

        return torch.cat(new_posteriors, dim=0)

    def _progressive_multiply(self, partial_dist: torch.Tensor, chunk_probs: torch.Tensor):
        # partial_dist: [B, C]
        # chunk_probs: [B, chunk_size, C]
        B, chunk_size, C = chunk_probs.shape
        out = partial_dist
        for w in range(chunk_size):
            out = out * chunk_probs[:, w, :]
            sums = out.sum(dim=1, keepdim=True).clamp_min(1e-40)
            out = out / sums
        return out

    def _update_temperatures(self, posteriors, steps=5):
        """
        We do gradient accumulation in micro-batches for items, so we don't OOM.
        """
        for _ in range(steps):
            # zero out old grad
            self.temp_optimizer.zero_grad()

            # We'll do partial forward+backward for all items in small item batches
            # then do one optimizer step after finishing them.
            total_log = 0.0
            T, W, C = self.dataset_tensor.shape

            # We'll accumulate in self.temperatures.grad
            for start_item in range(0, T, self.item_batch):
                end_item = min(start_item + self.item_batch, T)
                # forward pass for items [start_item:end_item]
                # partial negative log-likelihood
                partial_neg_log = self._calc_partial_neglog(
                    start_item, end_item
                )
                # we scale partial_neg_log by (1 / num_batches) or just do additive
                # Typically you might do partial_neg_log / (T / self.item_batch) to average,
                # but it’s optional. We'll just accumulate.
                partial_neg_log.backward()  # accumulate grads
                total_log += partial_neg_log.item()

            # Now we've accumulated gradients for all item micro-batches, do one step
            self.temp_optimizer.step()
            with torch.no_grad():
                self.temperatures.clamp_(min=1e-3, max=50.0)

    def _calc_partial_neglog(self, start_item, end_item):
        """
        Build the partial-dist for items in [start_item, end_item],
        do the progressive multiplication, then compute -log( sum_c [ prior * product ] ).
        Return a scalar that we can backward() on.
        """
        device = self.dataset_tensor.device
        B = end_item - start_item
        T, W, C = self.dataset_tensor.shape
        partial_dist = torch.ones((B, C), device=device, requires_grad=True)

        widx = 0
        while widx < W:
            wend = min(widx + self.worker_chunk, W)
            chunk_logits = self.dataset_tensor[start_item:end_item, widx:wend, :]
            chunk_scaled = torch.einsum('bwc,w->bwc', chunk_logits, 1.0 / self.temperatures[widx:wend])
            chunk_probs = F.softmax(chunk_scaled, dim=2).clamp_min(1e-40)

            partial_dist = self._progressive_multiply(partial_dist, chunk_probs)
            widx = wend

        partial_dist = partial_dist * self.class_prior.unsqueeze(0)
        sum_c = partial_dist.sum(dim=1).clamp_min(1e-40)
        # negative log-likelihood for these items
        neg_log = -torch.log(sum_c).sum()
        return neg_log

    @torch.no_grad()
    def _calc_log_likelihood_no_grad(self):
        """
        Purely for logging. We do a no-grad approach to measure total log-likelihood quickly.
        """
        T, W, C = self.dataset_tensor.shape
        logL = 0.0

        for start_item in range(0, T, self.batch_size):
            end_item = min(start_item + self.batch_size, T)
            B = end_item - start_item

            partial_dist = torch.ones((B, C), device=self.dataset_tensor.device)
            widx = 0
            while widx < W:
                wend = min(widx + self.worker_chunk, W)
                chunk_logits = self.dataset_tensor[start_item:end_item, widx:wend, :]
                chunk_scaled = torch.einsum('bwc,w->bwc', chunk_logits, 1.0 / self.temperatures[widx:wend])
                chunk_probs = F.softmax(chunk_scaled, dim=2).clamp_min(1e-40)

                # progressive multiply
                for w in range(chunk_probs.shape[1]):
                    partial_dist *= chunk_probs[:, w, :]
                    sums = partial_dist.sum(dim=1, keepdim=True).clamp_min(1e-40)
                    partial_dist /= sums

                widx = wend

            partial_dist *= self.class_prior.unsqueeze(0)
            sum_c = partial_dist.sum(dim=1).clamp_min(1e-40)
            batch_ll = torch.log(sum_c).sum().item()
            logL += batch_ll

        return logL

import torch
import torch.nn.functional as F
import math
import logging

class BinnedDawidSkene:
    def __init__(self,
                 class_num,
                 max_iter=100,
                 tolerance=0.01,
                 batch_size=1000) -> None:
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size  # Adjust if memory is limited

    def run(self, dataset):
        """
        dataset: A FloatTensor of shape (T, W, C)
            - T = number of tasks
            - W = number of workers
            - C = number of classes
          Each dataset[t, w, :] is the post-softmax probabilities from worker w on task t.
        
        We'll build:
          confidence_bin[t, w] ∈ {0,1,2}
             (depending on max probability <0.33, <0.67, or >=0.67)
        Then hold 3 confusion matrices for each worker:
          error_rates[w, bin, actual_class, reported_class]
        """
        device = dataset.device
        self.device = device
        self.task_num, self.worker_num, _ = dataset.shape
        self.dataset_tensor = dataset
        T, W, C = dataset.shape

        # ---------------------------------------------------------------------
        # Step 1. Assign each (task, worker) pair to a confidence bin {0,1,2}.
        # ---------------------------------------------------------------------
        max_scores, _ = self.dataset_tensor.max(dim=2)  # shape [T, W]
        confidence_bin = torch.full_like(max_scores, fill_value=-1, dtype=torch.long)
        confidence_bin[max_scores < 0.33] = 0
        confidence_bin[(max_scores >= 0.33) & (max_scores < 0.67)] = 1
        confidence_bin[max_scores >= 0.67] = 2
        # We'll keep this around so both M-step and E-step can see it
        self.confidence_bin = confidence_bin

        # ---------------------------------------------------------------------
        # Initialize predict_label:
        #   predict_label[t] is a vector of length C that represents the
        #   "soft" distribution over the true class of task t.
        #
        # A common choice is to start from the average across workers
        # (i.e. if each row was previously 1-hot). In the case of post-softmax
        # predictions, a simple start is:
        #   average across workers => shape [T, C].
        # Then re-normalize across C so each row sums to 1.
        # ---------------------------------------------------------------------
        mean_over_workers = self.dataset_tensor.mean(dim=1)  # [T, C]
        predict_label = mean_over_workers / (mean_over_workers.sum(dim=1, keepdim=True) + 1e-10)

        # EM loop:
        flag = True
        prev_error_rates, prev_predict_label = None, None
        iter_num = 0

        while flag:
            # M-step: update confusion matrices
            error_rates = self._m_step(predict_label)

            # E-step: update predict_label
            next_predict_label = self._e_step(predict_label, error_rates)

            # Evaluate log-likelihood for debugging
            log_L = self._get_likelihood(predict_label, error_rates)

            if iter_num == 0:
                logging.info("{}\t{}".format(iter_num, log_L))
            else:
                # Check convergence
                marginal_predict = torch.sum(predict_label, 0) / self.task_num
                prev_marginal_predict = torch.sum(prev_predict_label, 0) / self.task_num
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

                print(f"Iteration {iter_num}: marginals_diff={marginals_diff.item():.5f} "
                      f"error_rates_diff={error_rates_diff.item():.5f} log_L={log_L:.5f}")

            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iter_num += 1

        # Once converged, compute final marginal label distribution:
        marginal_predict = torch.sum(predict_label, 0) / self.task_num

        # Compute "worker reliability" as the sum of diagonal for each of the 3 confusion matrices
        # weighted by how often each bin is used. One approach:
        # --------------------------------------------------------------------
        # 1) Count how many tasks fall into each bin for each worker: frac_uses[w,b]
        #    => shape [W,3]
        #    This is a per-worker bin usage count, then divide by T => usage fraction
        # --------------------------------------------------------------------
        # confidence_bin: shape [T, W], each entry ∈ {0,1,2}
        # We want a count over tasks for each worker w in bin b.
        # "torch.nn.functional.one_hot(confidence_bin, num_classes=3)" => shape [T,W,3]
        # then sum over tasks dimension => shape [W,3]
        # We just need to permute or rearrange dimensions properly.

        bin_one_hot = torch.nn.functional.one_hot(confidence_bin, num_classes=3).float() 
        # shape [T, W, 3]

        # sum over tasks (dim=0), leaving shape [W,3]
        bin_usage_counts = bin_one_hot.sum(dim=0)  
        # fraction of tasks in each bin
        bin_usage_fraction = bin_usage_counts / float(T)

        # --------------------------------------------------------------------
        # 2) Probability of correctness per bin: 
        #    sum_i( marginal_predict[i] * error_rates[w,b,i,i] )
        #    => shape [W,3]
        # --------------------------------------------------------------------
        # Diagonal slice: shape [W,3,C]
        diag_ = torch.diagonal(error_rates, offset=0, dim1=2, dim2=3)
        # 'diag_' now has error_rates[w,b,i,i] along the last dimension i

        # We want sum over i of [marginal_predict[i] * diag_[w,b,i]].
        # => prob_correct[w,b] = ∑_{i} p_i * confusion[w,b,i,i]
        # That is a matrix multiplication over dimension i:
        # diag_: [W,3,C]
        # marginal_predict: [C]
        # => resulting shape [W,3]
        prob_correct = torch.einsum('wbc,c->wb', diag_, marginal_predict)

        # --------------------------------------------------------------------
        # 3) Weighted average: reliability[w] = ∑_{b} bin_usage_fraction[w,b] * prob_correct[w,b]
        #    => shape [W]
        # --------------------------------------------------------------------
        reliability = torch.einsum('wb,wb->w', bin_usage_fraction, prob_correct)

        return marginal_predict, error_rates, reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return ((marginals_diff < self.tolerance and error_rates_diff < self.tolerance) 
                or iter_num > self.max_iter)

    # -------------------------------------------------------------------------
    # M-step: Recompute "error_rates" = [W, 3, C, C].
    #
    # We want:
    #   error_rates[w, b, actual_class, reported_class] = 
    #       \sum_{t} predict_label[t, actual_class] * dataset[t, w, reported_class]
    #          * 1{confidence_bin[t,w] = b}
    #    / (normalization across reported_class)
    # 
    # We do that in chunks of "batch_size" workers for memory reasons.
    # -------------------------------------------------------------------------
    def _m_step(self, predict_label):
        device = predict_label.device
        epsilon = 1e-10

        # We'll store 3 confusion matrices per worker: shape [W, 3, C, C]
        error_rates = torch.zeros((self.worker_num, 3, self.class_num, self.class_num),
                                  dtype=torch.float32, device=device)

        # For memory reasons, do chunks of workers:
        for start in range(0, self.worker_num, self.batch_size):
            end = min(start + self.batch_size, self.worker_num)

            # shape: [T, (end-start), C]
            batch_dataset = self.dataset_tensor[:, start:end, :]
            # shape: [T, (end-start)]
            batch_conf_bin = self.confidence_bin[:, start:end]

            # 1) Make a one-hot version of the confidence bin:
            #    shape => [T, (end-start), 3]
            bin_onehot = F.one_hot(batch_conf_bin, num_classes=3).float()

            # 2) Compute partial confusion counts by summing:
            #    sum_{t} predict_label[t,i] * batch_dataset[t, w, c] * bin_onehot[t, w, b]
            #    We'll do this in a single einsum:
            #
            #      'ti, twc, twb -> wbic'
            #      t = task, i = actual_class,
            #      w = worker_in_this_batch, c = reported_class, b = confidence_bin
            #
            #    The result is shape [ (end-start), 3, C, C ] => 
            #          (for each worker in the batch, each bin, each actual_class i, each reported_class c)
            worker_error_rate = torch.einsum(
                'ti, twc, twb -> wbic',
                predict_label, 
                batch_dataset, 
                bin_onehot
            )
            # worker_error_rate has shape [batch_size, 3, C, C]

            # 3) Normalize across reported_class dimension to get probabilities
            #    sum over c in [C], keep dim
            sum_over_reported = worker_error_rate.sum(dim=3, keepdim=True) + epsilon
            worker_error_rate = worker_error_rate / sum_over_reported

            # Store these results in error_rates
            error_rates[start:end] = worker_error_rate

        return error_rates

    # -------------------------------------------------------------------------
    # E-step:
    #    next_predict_label[t, k] ∝ prior[k] * product_over_w( P( w's observed | actual=k ) )
    #
    # But now the probability for worker w uses the bin-specific confusion matrix:
    #      P_w( observed | actual=k )
    #        = sum_{reported_class} [ dataset[t, w, reported_class] * error_rates[w, bin, k, reported_class] ]
    #      where bin = confidence_bin[t,w]
    #
    # We'll do a loop over workers w, but vectorize the multiplication across tasks t.
    # -------------------------------------------------------------------------
    def _e_step(self, predict_label, error_rates):
        device = predict_label.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        # prior[k]
        marginal_probability = torch.sum(predict_label, dim=0) / T  # shape [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)  # shape [C]

        # We'll use log_pi = log( error_rates ), shape [W,3,C,C]
        log_pi = torch.log(error_rates + 1e-10)

        next_predict_label = torch.zeros([T, C], dtype=torch.float32, device=device)

        # Process tasks in batches so we don't blow up memory
        batch_size = self.batch_size
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)

            # shape: [batch_size, W, C]
            batch_dataset = self.dataset_tensor[start:end]  
            # shape: [batch_size, W]
            batch_conf_bin = self.confidence_bin[start:end]

            # We'll accumulate log_class_likelihood for each of the T' tasks in this batch, each of the C classes
            # shape [T', C]
            log_class_likelihood = torch.zeros((end - start, C), device=device)

            # We loop over each worker, partially vectorizing the contribution
            for w in range(W):
                # bin for each of the tasks in this batch, shape [T']
                b_idx = batch_conf_bin[:, w]  # each entry ∈ {0,1,2}

                # Gather the worker's log_pi => shape [3, C, C] for this w
                # We'll do an einsum to multiply dataset[t,w,c_reported] with log_pi[w, b, k, c_reported].
                # Steps:
                #   1) "temp = torch.einsum('tc, bkc->tbk', batch_dataset[:, w], log_pi[w])"
                #      BUT we must index the dimension 'b' = 3. So log_pi[w] has shape [3, C, C].
                #      This yields shape [T', 3, C], where
                #         temp[t, b, k] = sum_c( batch_dataset[t, w, c]*log_pi[w, b, k, c] )
                #
                #   2) We select the correct bin per task t => temp[t, b_idx[t], :] => shape [T', C].
                #
                #   3) We add that to log_class_likelihood[t].
                #
                worker_log_pi = log_pi[w]  # [3, C, C]
                # shape [T', 3, C]
                temp = torch.einsum('tc,bkc->tbk', batch_dataset[:, w], worker_log_pi)
                # Now pick out the correct bin for each row
                # shape [T', C]
                chosen_bins = temp[range(end - start), b_idx, :]
                # Accumulate
                log_class_likelihood += chosen_bins

            # Now log_class_likelihood[t,k] = sum over w of log( P_{w,bin} ) in the sense that we
            # haven't done a log-sum, but *we have sums of logs only if worker probabilities were log.
            # Actually, we used log_pi for the confusion part, but the multiplication by dataset[t,w]
            # was done in normal space -> we took an einsum. So 'temp' is sum_{reported} p_reported * log_pi.
            #
            # Typically you'd want
            #   log_class_likelihood[t,k] = sum_over_w( log( sum_{c} dataset[t,w,c] * pi[w,bin,k,c] ) )
            # The code above merges partial sums. So if you want everything purely in log, you'd do a safe log-sum-exp.
            # For brevity, let's treat `log_class_likelihood` as the additive term. Then:
            #
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            # shape [T', C]
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)

            next_predict_label[start:end] = torch.exp(log_posterior)

        return next_predict_label

    # -------------------------------------------------------------------------
    # Compute likelihood (in log form) for debugging. Similar logic as E-step:
    # -------------------------------------------------------------------------
    def _get_likelihood(self, predict_label, error_rates):
        device = predict_label.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        # prior[k]
        marginal_probability = torch.sum(predict_label, dim=0) / T  # shape [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        log_pi = torch.log(error_rates + 1e-10)

        log_L = 0.0
        batch_size = self.batch_size
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_dataset = self.dataset_tensor[start:end]  
            batch_conf_bin = self.confidence_bin[start:end]

            # shape [T', C]
            log_class_likelihood = torch.zeros((end - start, C), device=device)

            for w in range(W):
                b_idx = batch_conf_bin[:, w]
                worker_log_pi = log_pi[w]  # [3, C, C]
                # shape [T', 3, C]
                temp = torch.einsum('tc,bkc->tbk', batch_dataset[:, w], worker_log_pi)
                # shape [T', C]
                chosen_bins = temp[range(end - start), b_idx, :]
                log_class_likelihood += chosen_bins

            # Now combine with the prior
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            # Sum of log-sum-exp over tasks
            log_L += torch.sum(torch.logsumexp(log_posterior, dim=1))

        return log_L.item()

import logging
import torch
import torch.nn.functional as F

class GatedBinnedDawidSkene:
    """
    A Dawid-Skene-style model with 3 bins per worker, 
    where the bin usage is governed by a learned gating function 
    of the worker's confidence (e.g. max predicted probability).
    """

    def __init__(self,
                 class_num,
                 max_iter=20,
                 tolerance=1e-3,
                 batch_size=1000,
                 gating_lr=0.1,
                 gating_steps=5):
        """
        Args:
          class_num: number of classes (C)
          max_iter: maximum EM iterations
          tolerance: stop if changes fall below this
          batch_size: used in M-step for memory scaling
          gating_lr: learning rate for gating parameter gradient steps
          gating_steps: how many gradient steps we do each M-step 
                        to update gating_params
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size

        # For the logistic gating:
        self.gating_lr = gating_lr
        self.gating_steps = gating_steps

    def run(self, dataset):
        """
        dataset: FloatTensor, shape [T, W, C]
          - T = number of tasks
          - W = number of workers (or models)
          - C = number of classes
          Each dataset[t, w, :] is the distribution over classes (e.g. softmax).

        Returns:
          - marginal_predict: shape [C], the overall fraction of each class
          - error_rates: shape [W, 3, C, C], confusion matrices per bin
          - reliability: shape [W], measure of worker reliability
          - predict_label: shape [T, C], posterior distribution over classes
        """
        device = dataset.device
        T, W, C = dataset.shape
        self.task_num = T
        self.worker_num = W
        self.dataset_tensor = dataset

        # 1) Initialize gating parameters: [W, 3, 2] => for each worker w, each bin b,
        #    we have (bias, slope). Start near zero.
        self.gating_params = torch.zeros((W, 3, 2), dtype=torch.float32, device=device)

        # 2) Initialize predict_label[t], shape [T, C], by averaging over workers
        mean_over_workers = dataset.mean(dim=1)  # [T, C]
        predict_label = mean_over_workers / (mean_over_workers.sum(dim=1, keepdim=True) + 1e-12)

        # 3) Initialize confusion matrices: error_rates[w,b,k,c]
        #    We'll start them near identity.
        error_rates = torch.zeros((W, 3, C, C), dtype=torch.float32, device=device)
        for w_i in range(W):
            for b in range(3):
                mat = torch.eye(C, device=device) + 0.01
                mat = mat / mat.sum(dim=1, keepdim=True)  # each row sums to 1
                error_rates[w_i, b] = mat

        # For convergence checking:
        prev_predict_label = predict_label.clone()
        prev_error_rates = error_rates.clone()

        # EM loop
        for iter_idx in range(self.max_iter):
            # E-step: produce new predict_label
            next_predict_label = self._e_step(predict_label, error_rates)

            # M-step part 1: update error_rates + get bin_resps[t,w,b]
            new_error_rates, bin_resps = self._m_step_error(next_predict_label, error_rates)

            # M-step part 2: update gating_params via bin_resps
            self._m_step_gating(bin_resps)

            # Evaluate (optional) log-likelihood
            log_L = self._get_likelihood(next_predict_label, new_error_rates)

            # Check convergence
            label_diff = (next_predict_label - prev_predict_label).abs().sum().item()
            err_diff = (new_error_rates - prev_error_rates).abs().sum().item()
            if label_diff < self.tolerance and err_diff < self.tolerance:
                print(f"Converged at iteration {iter_idx}, log_L={log_L:.5f}")
                break

            # Print iteration info
            print(f"Iteration {iter_idx}: label_diff={label_diff:.5g}, err_diff={err_diff:.5g}, log_L={log_L:.5f}")

            prev_predict_label = next_predict_label
            prev_error_rates = new_error_rates
            predict_label = next_predict_label
            error_rates = new_error_rates

        # Once done, compute final marginal distribution of classes
        marginal_predict = predict_label.mean(dim=0)  # shape [C]

        # Example reliability measure:
        reliability = self._compute_reliability(error_rates, marginal_predict)

        return marginal_predict, error_rates, reliability, predict_label

    # -------------------------------------------------------------------------
    # gating_probs: for a single worker w_i, compute pi_{t,b} for each item t
    # based on gating_params[w_i,b].
    # We'll define x_{t,w} = max predicted probability for the worker's distribution.
    # -------------------------------------------------------------------------
    def gating_probs(self, w_i, x):
        """
        Args:
          w_i: worker index
          x: shape [T], the max confidence for tasks t=0..T-1
        gating_params[w_i,b,:] = (bias, slope)
        Returns: pi shape [T, 3], each row is a softmax over bins
        """
        # gating_params[w_i,b,0] = bias
        # gating_params[w_i,b,1] = slope
        bias = self.gating_params[w_i, :, 0]   # shape [3]
        slope = self.gating_params[w_i, :, 1]  # shape [3]

        # logit[t,b] = bias[b] + slope[b]*x[t]
        logit = bias.view(1,3) + x.view(-1,1)*slope.view(1,3)
        return F.softmax(logit, dim=1)  # shape [T,3]

    # -------------------------------------------------------------------------
    # E-step:
    #
    #   next_predict_label[t,k] ∝ prior[k] * ∏_{w} sum_{b} [ gating_probs[w,b](x_{t,w}) *
    #                                                    product_{c}(error_rates[w,b,k,c]^ dataset[t,w,c]) ]
    #
    # We'll do this in log-space for stability and chunk over tasks if needed.
    # -------------------------------------------------------------------------
    def _e_step(self, prev_predict_label, error_rates):
        device = self.dataset_tensor.device
        T, W, C = self.task_num, self.worker_num, self.class_num

        # prior[k] = average of prev_predict_label across tasks
        prior = prev_predict_label.mean(dim=0) + 1e-12
        log_prior = torch.log(prior)

        next_predict_label = torch.zeros((T, C), dtype=torch.float32, device=device)

        # We'll split tasks in batches if T is large
        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            tsize = end - start

            # shape [tsize, C]
            batch_predict = torch.zeros((tsize, C), device=device)

            # sum_{w} log( sum_{b} gating_probs(...) * product_{c}(error[w,b,k,c]^p_{t,w,c}) )
            # We'll build a log_likelihood[t',k] for each task t'.
            log_lik = torch.zeros((tsize, C), device=device)

            for w_i in range(W):
                # get x_{t,w_i}, i.e. max confidence => shape [tsize]
                worker_slice = self.dataset_tensor[start:end, w_i, :]  # [tsize, C]
                x_w = worker_slice.max(dim=1).values  # shape [tsize]

                # gating_probs => shape [tsize, 3]
                p_bin = self.gating_probs(w_i, x_w)

                # log_error[w_i,b,k,c]
                # We'll precompute log_error because we do product_{c} => sum_{c} dataset[t,c]*log_error[w_i,b,k,c].
                log_error = torch.log(error_rates[w_i] + 1e-12)  # shape [3, C, C]

                # We'll do this for each bin in a vectorized way.
                # For each bin b, for each k, we want sum over c of worker_slice[t,c]*log_error[b,k,c].
                # => shape [tsize, 3, C]
                temp = torch.einsum('tc,bkc->tbk', worker_slice, log_error)

                # Now exponentiate + multiply by p_bin[t,b]:
                # shape [tsize, 3, C]
                exp_temp = torch.exp(temp)  # product_{c}, ignoring log base
                # multiply each bin dimension by p_bin:
                # shape [tsize,3, C]
                weighted = exp_temp * p_bin.unsqueeze(-1)

                # sum over b => shape [tsize, C]
                sum_b = weighted.sum(dim=1)

                # take log => shape [tsize, C]
                # add to log_lik => in log-space we do: log_lik += log(sum_b).
                # but we have to do log-sum: if we want to sum these across w, we do:
                #   log_lik[t,k] += log( sum_b[t,k] ).
                # We'll do:
                log_lik += torch.log(sum_b + 1e-12)

            # Now add the log_prior
            log_lik = log_lik + log_prior.view(1, C)

            # convert to posterior => exponentiate and normalize across k
            log_lik_norm = log_lik - torch.logsumexp(log_lik, dim=1, keepdim=True)
            batch_predict = torch.exp(log_lik_norm)

            next_predict_label[start:end] = batch_predict

        return next_predict_label

    # -------------------------------------------------------------------------
    # M-step (part 1): update error_rates.
    #
    # We need p(k,b| t,w).  We'll store partial results in bin_resps[t,w,b].
    #
    # p(k,b | t,w) = p(k|t) * [ gating_probs[w,b](x_{t,w}) * ∏_{c}(error[w,b,k,c]^{dataset[t,w,c]}) ]
    #               / normalizer_{b'}
    #
    # We'll sum over tasks to get new confusion matrix for each (w,b,k,c).
    # -------------------------------------------------------------------------
    def _m_step_error(self, predict_label, error_rates_old):
        device = self.dataset_tensor.device
        T, W, C = self.task_num, self.worker_num, self.class_num

        new_error = torch.zeros_like(error_rates_old)
        # denominators for confusion matrix: shape [W,3,C]
        denom = torch.zeros((W,3,C), dtype=torch.float32, device=device)

        # We'll also store bin_resps[t,w,b] = sum_{k} p(k,b| t,w).
        bin_resps = torch.zeros((T, W, 3), dtype=torch.float32, device=device)

        log_error_old = torch.log(error_rates_old + 1e-12)

        # For memory reasons, chunk over tasks
        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            tsize = end - start

            # shape [tsize, C]
            batch_plabel = predict_label[start:end]

            for w_i in range(W):
                # slice of dataset => shape [tsize, C]
                worker_slice = self.dataset_tensor[start:end, w_i, :]
                # gating => shape [tsize,3]
                x_w = worker_slice.max(dim=1).values  # scalar feature
                p_bin = self.gating_probs(w_i, x_w)   # [tsize,3]

                w_log_err = log_error_old[w_i]        # [3,C,C]

                # We'll do the partial computations:
                # temp[t,b,k] = p_bin[t,b] * exp( sum_c( worker_slice[t,c]* w_log_err[b,k,c] ) )
                temp = torch.einsum('tc,bkc->tbk', worker_slice, w_log_err)
                temp = torch.exp(temp)  # shape [tsize, 3, C]
                temp = temp * p_bin.unsqueeze(-1)  # [tsize,3,C]

                # multiply also by predict_label[t,k]
                # => p(k,b|t,w) = predict_label[t,k] * temp[t,b,k], up to normalization across b
                # We'll compute normalizer for b => shape [tsize, C]
                sum_b = temp.sum(dim=1)  # shape [tsize, C]
                sum_b = sum_b + 1e-12

                # now incorporate p(k|t) => shape [tsize, C]
                # We'll broadcast multiply: temp[t,b,k] *= predict_label[t,k]
                temp *= batch_plabel.unsqueeze(1)

                # p(k,b|t,w) = temp[t,b,k] / sum_b[t,k]
                # shape => [tsize, 3, C]
                post_bk = temp / sum_b.unsqueeze(1)

                # Summation to get new error rates:
                # new_error[w_i,b,k,c] += sum_t( post_bk[t,b,k]* dataset[t,w_i,c] )
                # We'll do an einsum: 'tbk, tC -> bkc' with sum over t
                # But careful: dataset[t,w_i,c] = worker_slice[t,c].
                # We'll do 'tbk,tc -> bkc' with sum over t
                contrib = torch.einsum('tbk,tc->bkc', post_bk, worker_slice)
                new_error[w_i] += contrib

                # denom[w_i,b,k] += sum_t( post_bk[t,b,k] * sum_c(worker_slice[t,c]) )
                # sum_c(worker_slice[t,c]) => shape [tsize]
                sum_over_c = worker_slice.sum(dim=1)  # [tsize]
                # we do 'tbk,t -> bk'
                denom_contrib = torch.einsum('tbk,t->bk', post_bk, sum_over_c)
                denom[w_i] += denom_contrib #.unsqueeze(-1)  # shape [b,k,1]

                # bin_resps[t,w_i,b] += sum_k( post_bk[t,b,k] )
                # => shape [tsize,3]
                # We'll sum over k
                sum_k = post_bk.sum(dim=2)  # [tsize,3]
                bin_resps[start:end, w_i, :] += sum_k

        # Now normalize error rates
        new_error_rates = new_error / (denom.unsqueeze(-1) + 1e-12)

        return new_error_rates, bin_resps

    def _m_step_gating(self, bin_resps):
        """
        bin_resps: FloatTensor, shape [T, W, 3],
        where bin_resps[t,w,b] = sum_k p(k,b | t,w) from the E-step.
        We'll do self.gating_steps iterations of gradient-based optimization
        of the gating parameters.

        We treat gating_params as shape [W,3,2], with requires_grad=True,
        and do one big forward pass each iteration, summing the cross-entropy 
        across all workers w and tasks t. Then backward once, then update.
        """
        device = self.dataset_tensor.device
        T, W, _ = bin_resps.shape

        # Make sure gating_params is a leaf with gradient tracking
        self.gating_params.requires_grad_(True)

        for step in range(self.gating_steps):
            # Zero out any gradient from the previous iteration
            if self.gating_params.grad is not None:
                self.gating_params.grad.zero_()

            # We'll accumulate the total loss across all workers and tasks
            total_loss = torch.tensor(0.0, device=device, requires_grad=False)

            # For memory reasons, we can do a loop over workers or tasks in mini-batches:
            # but for simplicity, let's just do a loop over all workers in one pass.
            for w_i in range(W):
                # shape [T, 3]: bin_resps for this worker
                w_bin_resp = bin_resps[:, w_i, :]  # soft labels
                # shape [T], x_{t,w_i} = max prob from dataset
                x_w = self.dataset_tensor[:, w_i, :].max(dim=1).values

                # We'll build the logits = gating_forward_all(...) 
                # This can be done by index into gating_params. 
                # Let's do it in a small sub-function:
                logit = self._gating_forward_all(w_i, x_w)  # shape [T,3]

                # Convert logits to log-softmax
                log_prob = F.log_softmax(logit, dim=1)  # [T,3]

                # Cross-entropy with "soft" labels w_bin_resp[t,b].
                # Negative log-likelihood = - sum_{t,b} [ w_bin_resp[t,b] * log_prob[t,b] ]
                loss_w = - (w_bin_resp * log_prob).sum()
                total_loss += loss_w

            # Now we have a scalar total_loss across all workers. 
            # Backprop once:
            total_loss.backward()

            # Gradient update: gating_params is shape [W,3,2]
            with torch.no_grad():
                self.gating_params -= self.gating_lr * self.gating_params.grad

        # Turn off grad for gating_params for safety
        self.gating_params.requires_grad_(False)


    def _gating_forward(self, w_i, x):
        """
        Return the *logits* for gating at worker w_i, not the softmax.
        This is for the internal gradient-based update. 
        gating_params[w_i,b] = [bias, slope].
        """
        if not hasattr(self.gating_params, 'grad') or self.gating_params.grad is None:
            self.gating_params.requires_grad_(True)

        bias = self.gating_params[w_i, :, 0]   # shape [3]
        slope = self.gating_params[w_i, :, 1]  # shape [3]
        # logit[t,b]
        logit = bias.view(1,3) + x.view(-1,1)*slope.view(1,3)
        return logit
    
    def _gating_forward_all(self, w_i, x):
        """
        Return the *logits* for gating at worker w_i over tasks t=0..T-1.
        gating_params[w_i, b, :] = (bias, slope).
        x: shape [T].
        Output shape: [T, 3].
        """
        # gating_params is shape [W,3,2]
        # gating_params[w_i] shape [3,2]
        # let bias = gating_params[w_i, :, 0]
        # let slope = gating_params[w_i, :, 1]
        bias = self.gating_params[w_i, :, 0]   # [3]
        slope = self.gating_params[w_i, :, 1]  # [3]

        # logit[t,b] = bias[b] + slope[b]* x[t]
        # shape => [T,3]
        logit = bias.view(1,3) + x.view(-1,1)*slope.view(1,3)
        return logit


    # -------------------------------------------------------------------------
    # log-likelihood for debugging (optional)
    # We'll do sum_{t} log( sum_{k} p(k| t) * bigTerm ), 
    # but more precisely we can do the usual "Q" function approach:
    # sum_{t,k} p(k|t)* [ sum_{w} log( sum_{b} gating_probs * ... ) + log prior[k] ].
    # We'll do a simpler approximate version for reference.
    # -------------------------------------------------------------------------
    def _get_likelihood(self, predict_label, error_rates):
        device = self.dataset_tensor.device
        T, W, C = self.task_num, self.worker_num, self.class_num

        # prior[k]
        prior = predict_label.mean(dim=0) + 1e-12
        log_prior = torch.log(prior)

        log_L = 0.0
        batch_size = self.batch_size

        log_err = torch.log(error_rates + 1e-12)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            tsize = end - start

            # shape [tsize, C]
            batch_plabel = predict_label[start:end]
            # build log_lik[t', k]
            log_lik = torch.zeros((tsize, C), device=device)

            for w_i in range(W):
                slice_ = self.dataset_tensor[start:end, w_i, :]  # [tsize, C]
                x_w = slice_.max(dim=1).values  # shape [tsize]
                p_bin = self.gating_probs(w_i, x_w)  # [tsize, 3]

                # sum_{b} p_bin[t,b]* exp( sum_c( slice_[t,c]* log_err[w_i,b,k,c] ) )
                w_log_err = log_err[w_i]  # [3, C, C]
                temp = torch.einsum('tc,bkc->tbk', slice_, w_log_err)  # [tsize,3,C]
                temp = torch.exp(temp)
                temp = temp * p_bin.unsqueeze(-1)
                sum_b = temp.sum(dim=1)  # [tsize, C]
                log_lik += torch.log(sum_b + 1e-12)

            log_lik += log_prior.view(1, C)
            # "Q function" style: sum_{t,k} p(k|t)* log_lik[t,k]
            ll_batch = (batch_plabel * log_lik).sum()
            log_L += ll_batch.item()

        return log_L

    # -------------------------------------------------------------------------
    # Just an example measure of "worker reliability":
    # Weighted average of diagonal in confusion matrices by how often each bin is used (on average).
    # Let's do:
    #  1) compute average gating_probs across tasks => bin usage
    #  2) compute probability of correct for each bin b => sum_{k} p(k)* error_rates[w,b,k,k]
    #  3) weighted sum => reliability[w].
    # -------------------------------------------------------------------------
    def _compute_reliability(self, error_rates, marginal_predict):
        W = self.worker_num
        C = self.class_num

        # diagonal slice => error_rates[w,b,k,k]
        diag_ = torch.diagonal(error_rates, offset=0, dim1=2, dim2=3)  # shape [W,3,C]

        # prob_correct_per_bin[w,b] = sum_{k} marginal_predict[k] * diag_[w,b,k]
        prob_correct_per_bin = torch.einsum('wbc,c->wb', diag_, marginal_predict)

        # approximate bin usage => average gating_probs across tasks
        # We'll just do x_{t,w} = max prob, gating_probs => average over t
        # shape => [W,3]
        usage = self._average_bin_usage()

        # reliability[w] = sum_b usage[w,b]* prob_correct_per_bin[w,b]
        reliability = torch.einsum('wb,wb->w', usage, prob_correct_per_bin)
        return reliability

    def _average_bin_usage(self):
        """
        For each worker w, compute average of gating_probs(w, x_{t,w}) across tasks t.
        We'll just do a simple approach: for each t, we take x_{t,w}, get gating_probs, 
        then average. This can be done in mini-batches if T is huge.
        Returns shape [W,3].
        """
        T, W, C = self.dataset_tensor.shape
        usage = torch.zeros((W,3), dtype=torch.float32, device=self.dataset_tensor.device)

        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            for w_i in range(W):
                slice_ = self.dataset_tensor[start:end, w_i, :]
                x_w = slice_.max(dim=1).values
                probs = self.gating_probs(w_i, x_w)  # shape [batch_size,3]
                usage[w_i] += probs.sum(dim=0)

        usage = usage / float(T)
        return usage


import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkerConfusionNet(nn.Module):
    """
    Small MLP that returns *raw logits* of shape [batch_size, C, C].
    We'll apply log-softmax outside this module to avoid large intermediate matrices.
    """

    def __init__(self, C, hidden_dim=16):
        super().__init__()
        self.C = C
        self.hidden_dim = hidden_dim

        if hidden_dim > 0:
            self.fc1 = nn.Linear(C, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, C*C)
        else:
            self.fc2 = nn.Linear(C, C*C)

    def forward(self, x):
        """
        x: shape [batch_size, C]
        returns: raw_logits of shape [batch_size, C, C].
        No softmax done here!
        """
        if self.hidden_dim > 0:
            h = F.relu(self.fc1(x))
            logits = self.fc2(h)  # [batch_size, C*C]
        else:
            logits = self.fc2(x)  # [batch_size, C*C]
        return logits.view(-1, self.C, self.C)  # [batch_size, C, C]

class NeuralDawidSkeneDoubleChunkedLogits(nn.Module):
    """
    Memory-optimized Neural Dawid–Skene that:
      - chunk over tasks
      - chunk over workers
      - use log-softmax *directly* on worker logits (no large mats + log(...) step)
      - optional AMP (mixed precision)
    """

    def __init__(
        self,
        class_num,
        worker_num,
        hidden_dim=16,
        max_iter=20,
        tolerance=1e-3,
        lr=1e-2,
        m_steps=5,
        task_batch_size=512,
        worker_batch_size=None,
        mixed_precision=True,
        device='cuda'
    ):
        """
        Args:
          class_num: C
          worker_num: W
          hidden_dim: MLP dimension (0 => single-layer)
          max_iter: EM iterations
          tolerance: stopping threshold
          lr: learning rate
          m_steps: # of gradient steps each EM iteration
          task_batch_size: chunk size for tasks T
          worker_batch_size: chunk size for workers W (None => process all W at once)
          mixed_precision: use AMP
          device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.C = class_num
        self.W = worker_num
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.lr = lr
        self.m_steps = m_steps
        self.task_batch_size = task_batch_size
        self.worker_batch_size = worker_batch_size
        if self.worker_batch_size is None:
            self.worker_batch_size = self.W
        self.mixed_precision = mixed_precision
        self.device = device

        # Create a net for each worker that returns raw logits
        self.worker_nets = nn.ModuleList([
            WorkerConfusionNet(self.C, self.hidden_dim) for _ in range(self.W)
        ])

    def run(self, dataset):
        """
        dataset: [T, W, C] FloatTensor
        Returns:
          marginal_predict: [C]
          final_error_rates: [W, C, C]
          reliability: [W]
          predict_label: [T, C]
        """
        self.to(self.device)
        dataset = dataset.to(self.device)

        T, W, C = dataset.shape
        assert W == self.W and C == self.C, "dataset shape mismatch"

        # Initialize predict_label by averaging each task's distribution
        predict_label = dataset.mean(dim=1)
        predict_label = predict_label / (predict_label.sum(dim=1, keepdim=True) + 1e-12)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        prev_predict_label = predict_label.clone()

        for em_iter in range(self.max_iter):
            # E-step
            predict_label_new = self._e_step(predict_label, dataset)

            # M-step
            self._m_step(predict_label_new, dataset)

            # check convergence
            label_diff = (predict_label_new - prev_predict_label).abs().sum().item()
            print(f"[EM Iter {em_iter}] label_diff={label_diff:.6g}")
            if label_diff < self.tolerance:
                print(f"Converged at iteration {em_iter}, label_diff={label_diff:.6g}")
                predict_label = predict_label_new
                break

            predict_label = predict_label_new
            prev_predict_label = predict_label_new

        # Summarize final confusion by averaging each worker's output
        final_error_rates = self._average_confusion_mats(dataset)
        marginal_predict = predict_label.mean(dim=0)
        # reliability = average diagonal
        reliability = final_error_rates.diagonal(dim1=1, dim2=2).mean(dim=1)
        return marginal_predict, final_error_rates, reliability, predict_label

    @torch.no_grad()
    def _e_step(self, predict_label, dataset):
        """
        E-step with chunking tasks & workers, using log-softmax of raw logits
        next_predict_label[t,k] ∝ prior[k] * product_{w} exp( sum_c x_{t,w}[c]* log_mats[t,k,c] ).
        We'll do it in log-space, chunk over tasks & workers.
        """
        T, W, C = dataset.shape
        prior = predict_label.mean(dim=0) + 1e-12
        log_prior = torch.log(prior)

        next_pred = torch.zeros((T, C), device=self.device)

        for t_start in range(0, T, self.task_batch_size):
            t_end = min(t_start + self.task_batch_size, T)
            size_ = t_end - t_start

            # log_lik[t',k]
            log_lik = torch.zeros((size_, C), device=self.device)

            for w_start in range(0, W, self.worker_batch_size):
                w_end = min(w_start + self.worker_batch_size, W)

                # do partial forward in mixed precision
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    for w_i in range(w_start, w_end):
                        # slice => [size_, C]
                        slice_ = dataset[t_start:t_end, w_i, :]

                        # raw logits => [size_, C, C]
                        logits = self.worker_nets[w_i](slice_)
                        # row log-softmax => shape [size_, C, C]
                        log_mats = F.log_softmax(logits, dim=2)

                        # worker_term[t,k] = sum_c slice_[t,c]* log_mats[t,k,c]
                        worker_term = torch.einsum('bc,bkc->bk', slice_, log_mats)

                        log_lik += worker_term

                        # free memory
                        del slice_, logits, log_mats, worker_term
                torch.cuda.empty_cache()

            # add log_prior
            log_lik += log_prior.view(1, C)

            # exponentiate + normalize across k
            max_ = torch.logsumexp(log_lik, dim=1, keepdim=True)
            log_lik = log_lik - max_
            post = torch.exp(log_lik)

            next_pred[t_start:t_end] = post

            del log_lik, post
            torch.cuda.empty_cache()

        return next_pred

    def _m_step(self, predict_label, dataset):
        """
        M-step: fix predict_label, do a few gradient steps chunking over tasks & workers.
        """
        T, W, C = dataset.shape

        for step_idx in range(self.m_steps):
            self.optimizer.zero_grad()
            total_ll = torch.tensor(0.0, device=self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                for t_start in range(0, T, self.task_batch_size):
                    t_end = min(t_start + self.task_batch_size, T)
                    size_ = t_end - t_start

                    batch_post = predict_label[t_start:t_end]
                    sum_chunk = torch.tensor(0.0, device=self.device)

                    for w_start in range(0, W, self.worker_batch_size):
                        w_end = min(w_start + self.worker_batch_size, W)

                        for w_i in range(w_start, w_end):
                            slice_ = dataset[t_start:t_end, w_i, :]
                            logits = self.worker_nets[w_i](slice_)
                            log_mats = F.log_softmax(logits, dim=2)

                            worker_term = torch.einsum('bc,bkc->bk', slice_, log_mats)
                            worker_sum = (worker_term * batch_post).sum(dim=1)  # shape [size_]

                            sum_chunk = sum_chunk + worker_sum.sum()

                            # free memory
                            del slice_, logits, log_mats, worker_term, worker_sum
                        torch.cuda.empty_cache()

                    total_ll = total_ll + sum_chunk
                    del batch_post, sum_chunk
                    torch.cuda.empty_cache()

            neg_loss = - total_ll
            self.scaler.scale(neg_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            del total_ll, neg_loss
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _average_confusion_mats(self, dataset):
        """
        Summarize by averaging each worker's confusion across all tasks.
        We'll do chunking & log-softmax -> softmax for each row.
        """
        T, W, C = dataset.shape
        conf_sum = torch.zeros((W, C, C), device=self.device)
        counts = torch.zeros((W,), device=self.device)

        for w_start in range(0, W, self.worker_batch_size):
            w_end = min(w_start + self.worker_batch_size, W)
            for w_i in range(w_start, w_end):
                total = torch.zeros((C, C), device=self.device)
                ccount = 0
                for t_start in range(0, T, self.task_batch_size):
                    t_end = min(t_start + self.task_batch_size, T)
                    slice_ = dataset[t_start:t_end, w_i, :]
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        logits = self.worker_nets[w_i](slice_)
                        # row softmax => [batch, C, C]
                        mats = F.softmax(logits, dim=2)
                    total += mats.sum(dim=0)
                    ccount += (t_end - t_start)
                    del slice_, logits, mats
                    torch.cuda.empty_cache()
                conf_sum[w_i] = total / float(ccount)
                counts[w_i] = ccount

        return conf_sum.cpu()


# class ConfidenceAdaptiveDS(DawidSkeneModel):
#     def __init__(self, class_num, confidence_bins=3, max_iter=100, 
#                  tolerance=0.01, batch_size=1000, alpha=0.1):
#         super().__init__(class_num, max_iter, tolerance, batch_size)
#         self.confidence_bins = confidence_bins
#         self.alpha = alpha

#     def _bin_confidence(self, confidences):
#         """Discretize confidence per-worker with adaptive thresholds"""
#         bins = torch.zeros_like(confidences, dtype=torch.long)
#         for w in range(confidences.size(1)):
#             worker_conf = confidences[:, w]
#             valid = worker_conf > 0  # Only consider actual annotations
#             if valid.sum() == 0:
#                 continue
                
#             # Compute quantiles per worker
#             quantiles = torch.quantile(
#                 worker_conf[valid],
#                 torch.linspace(0, 1, self.confidence_bins+1, device=confidences.device)
#             )
#             bins[valid, w] = torch.bucketize(worker_conf[valid], quantiles)
            
#         return bins

#     def _m_step(self, predict_label):
#         epsilon = 1e-10
#         error_rates = torch.zeros((self.worker_num, self.confidence_bins, 
#                                  self.class_num, self.class_num),
#                                 device=predict_label.device)
        
#         confidences = self.dataset_tensor.max(dim=2)[0]  # [N, W]
#         bins = self._bin_confidence(confidences)  # [N, W]

#         for w in range(self.worker_num):
#             worker_mask = (self.dataset_tensor[:, w].sum(dim=1) > 0)  # [N]
#             worker_bins = bins[:, w]  # [N]
            
#             for b in range(self.confidence_bins):
#                 bin_mask = (worker_bins == b) & worker_mask
#                 if bin_mask.sum() == 0:
#                     error_rates[w, b] = torch.eye(self.class_num, device=error_rates.device)
#                     continue
                
#                 bin_probs = predict_label[bin_mask]  # [T_bin, C]
#                 bin_obs = self.dataset_tensor[bin_mask, w]  # [T_bin, C]
                
#                 numerator = torch.einsum('tc,to->co', bin_probs, bin_obs)
#                 denominator = bin_probs.sum(dim=0).unsqueeze(1) + epsilon
                
#                 eye = torch.eye(self.class_num, device=error_rates.device)
#                 error_rates[w, b] = (numerator + self.alpha * eye) / (denominator + self.alpha)

#         return error_rates


#     def _e_step(self, predict_label, error_rates):
#         confidences = self.dataset_tensor.max(dim=2)[0]  # [N, W]
#         bins = self._bin_confidence(confidences)  # [N, W]
        
#         log_pi = torch.log(error_rates + 1e-10)  # [W, B, C, C]
#         next_predict_label = torch.zeros_like(predict_label)
        
#         for n in range(self.task_num):
#             worker_mask = self.dataset_tensor[n].sum(dim=1) > 0
#             w_indices = torch.where(worker_mask)[0]
#             b_indices = bins[n, worker_mask]
            
#             # Gather relevant confusion matrices using advanced indexing
#             log_probs = log_pi[w_indices, b_indices]  # [active_workers, C, C]
#             annotations = self.dataset_tensor[n, worker_mask]  # [active_workers, C]
            
#             # Compute log likelihood contribution from each worker
#             log_likelihood = torch.einsum('wc,wc->w', annotations, 
#                                         torch.logsumexp(log_probs, dim=2))
            
#             # Combine with prior
#             log_posterior = torch.log(predict_label[n] + 1e-10) + log_likelihood.sum()
#             next_predict_label[n] = torch.exp(log_posterior - torch.logsumexp(log_posterior, dim=0))
            
#         return next_predict_label

#     def get_worker_reliability(self, error_rates):
#         """Calculate comprehensive reliability score considering confidence patterns"""
#         # 1. Base accuracy weighted by confidence
#         diag = torch.diagonal(error_rates, dim1=2, dim2=3)  # [W, B, C]
#         class_weights = self.dataset_tensor.mean(dim=0).sum(dim=1)  # [W, C]
#         confidence_weights = error_rates.mean(dim=3).max(dim=2)[0]  # [W, B]
        
#         # 2. Compute reliability per confidence bin
#         bin_reliability = (diag * class_weights.unsqueeze(1)).sum(dim=2)  # [W, B]
        
#         # 3. Final score emphasizes high-confidence performance
#         reliability = (bin_reliability * confidence_weights).sum(dim=1)  # [W]
        
#         return {w: reliability[w].item() for w in range(self.worker_num)}

class ConfidenceAdaptiveDS(DawidSkeneModel):
    def __init__(self, class_num, confidence_bins=3, max_iter=100, 
                 tolerance=0.01, batch_size=1000, alpha=0.1):
        super().__init__(class_num, max_iter, tolerance, batch_size)
        self.confidence_bins = confidence_bins
        self.alpha = alpha

    def _bin_confidence(self, confidences):
        """Discretize confidence per-worker with adaptive thresholds and clamping"""
        bins = torch.zeros_like(confidences, dtype=torch.long)
        for w in range(confidences.size(1)):
            worker_conf = confidences[:, w]
            valid = worker_conf > 0  # Only consider actual annotations
            if valid.sum() < 2:  # Need at least 2 points to compute quantiles
                continue
                
            # Compute quantiles with slight epsilon adjustment
            quantiles = torch.quantile(
                worker_conf[valid],
                torch.linspace(0, 1, self.confidence_bins+1, device=confidences.device)
            )
            # Ensure upper bound exceeds max confidence
            quantiles[-1] += 1e-6  
            
            worker_bins = torch.bucketize(worker_conf[valid], quantiles, right=True)
            # Clamp bins to valid range
            worker_bins = torch.clamp(worker_bins, 0, self.confidence_bins-1)
            bins[valid, w] = worker_bins
            
        return bins

    def _m_step(self, predict_label):
        epsilon = 1e-10
        error_rates = torch.zeros((self.worker_num, self.confidence_bins, 
                                 self.class_num, self.class_num),
                                device=predict_label.device)
        
        confidences = self.dataset_tensor.max(dim=2)[0]  # [N, W]
        bins = self._bin_confidence(confidences)  # [N, W]

        for w in range(self.worker_num):
            worker_mask = self.dataset_tensor[:, w].sum(dim=1) > 0  # [N]
            worker_bins = bins[:, w]  # [N]
            
            for b in range(self.confidence_bins):
                bin_mask = (worker_bins == b) & worker_mask
                if bin_mask.sum() == 0:
                    # Regularize with uniform distribution instead of identity
                    error_rates[w, b] = torch.ones(self.class_num, self.class_num, 
                                                 device=error_rates.device) / self.class_num
                    continue
                
                bin_probs = predict_label[bin_mask]  # [T_bin, C]
                bin_obs = self.dataset_tensor[bin_mask, w]  # [T_bin, C]
                
                numerator = torch.einsum('tc,to->co', bin_probs, bin_obs)
                denominator = bin_probs.sum(dim=0).unsqueeze(1) + epsilon
                
                # Add regularization to prevent NaN
                error_rates[w, b] = (numerator + self.alpha) / (denominator + self.alpha * self.class_num)

        return error_rates

    def _e_step(self, predict_label, error_rates):
        confidences = self.dataset_tensor.max(dim=2)[0]  # [N, W]
        bins = self._bin_confidence(confidences)  # [N, W]
        
        log_pi = torch.log(error_rates + 1e-10)  # [W, B, C, C]
        next_predict_label = torch.zeros_like(predict_label)
        
        for n in range(self.task_num):
            worker_mask = self.dataset_tensor[n].sum(dim=1) > 0  # [W]
            active_workers = torch.where(worker_mask)[0]
            if len(active_workers) == 0:
                next_predict_label[n] = predict_label[n]  # No change if no annotations
                continue
                
            worker_bins = bins[n, active_workers]  # [active_workers]
            
            # Validate bin indices are within bounds
            worker_bins = torch.clamp(worker_bins, 0, self.confidence_bins-1)
            
            # Gather confusion matrices for active workers and their bins
            log_probs = log_pi[active_workers, worker_bins]  # [active_workers, C, C]
            annotations = self.dataset_tensor[n, active_workers]  # [active_workers, C]
            
            # Compute log likelihood using matrix operations
            log_likelihood = torch.einsum('wc,wck->k', annotations, log_probs)
            log_posterior = torch.log(predict_label[n] + 1e-10) + log_likelihood
            log_posterior -= torch.logsumexp(log_posterior, dim=0)
            
            next_predict_label[n] = torch.exp(log_posterior)

        return next_predict_label

    def get_worker_reliability(self, error_rates):
        """Calculate reliability using trace of confusion matrices weighted by usage"""
        reliability = torch.zeros(self.worker_num, device=error_rates.device)
        for w in range(self.worker_num):
            # Weight each bin by its usage frequency
            bin_weights = (self.dataset_tensor[:, w].max(dim=1)[0] > 0).float().mean(dim=0)
            weighted_trace = (error_rates[w].diagonal(dim1=1, dim2=2) * bin_weights).sum()
            reliability[w] = weighted_trace
        return {w: reliability[w].item() for w in range(self.worker_num)}
    
    def _get_likelihood(self, predict_label, error_rates):
        marginal_probability = torch.sum(predict_label, 0) / self.task_num  # [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)  # [C]
        log_pi = torch.log(error_rates + 1e-10)  # [W, B, C, C]
        
        # Get confidence scores and bins
        confidences = self.dataset_tensor.max(dim=2)[0]  # [N, W]
        bins = self._bin_confidence(confidences)  # [N, W]
        
        log_L = 0
        batch_size = self.batch_size
        for start in range(0, self.task_num, batch_size):
            end = min(start + batch_size, self.task_num)
            batch_dataset = self.dataset_tensor[start:end]  # [batch_size, W, C]
            batch_bins = bins[start:end]  # [batch_size, W]
            
            # Gather relevant confusion matrices for each worker and example
            log_probs = log_pi.unsqueeze(0)[  # [1, W, B, C, C]
                torch.arange(end-start)[:, None, None],  # Batch dimension
                torch.arange(self.worker_num)[None, :, None],  # Worker dimension
                batch_bins[:, :, None]  # Bin dimension
            ]  # [batch_size, W, 1, C, C]
            
            # Compute log likelihood
            log_class_likelihood = torch.einsum(
                'twc,twck->tk', 
                batch_dataset, 
                log_probs.squeeze(2)  # [batch_size, W, C, C]
            )
            
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_L += torch.sum(torch.logsumexp(log_posterior, dim=1))
            
        return log_L.item()


class ContinuousConfidenceDS(DawidSkeneModel):
    def __init__(self, class_num, worker_num, max_iter=100, tolerance=0.01, 
                 batch_size=1000, l2_reg=0.1):
        super().__init__(class_num, max_iter, tolerance, batch_size)
        self.worker_num = worker_num
        self.l2_reg = l2_reg
        torch.set_default_device('cuda')
        
        # Learnable parameters for confidence calibration
        self.calib_slope = nn.Parameter(torch.ones(self.worker_num))
        self.calib_bias = nn.Parameter(torch.zeros(self.worker_num))
        
    def _get_confusion_matrix(self, worker_id, confidence):
        """Continuous function mapping confidence to confusion matrix"""
        base_matrix = self.base_confusion[worker_id]  # [C, C]
        ideal_matrix = torch.eye(self.class_num)  # [C, C]
        
        # Sigmoidal mixing based on calibrated confidence
        calibrated_conf = torch.sigmoid(self.calib_slope[worker_id] * confidence 
                                      + self.calib_bias[worker_id])
        return calibrated_conf * ideal_matrix + (1 - calibrated_conf) * base_matrix

    def _m_step(self, predict_label):
        # Update base confusion matrices with regularization
        epsilon = 1e-10
        base_confusion = torch.zeros((self.worker_num, self.class_num, self.class_num),
                                   device=predict_label.device)
        
        for w in range(self.worker_num):
            mask = self.dataset_tensor[:,w].sum(dim=1) > 0
            confidences = self.dataset_tensor[mask,w].max(dim=1)[0]
            
            # Calculate effective weights considering confidence
            weights = 1 - torch.sigmoid(self.calib_slope[w] * confidences 
                                      + self.calib_bias[w]).detach()
            
            numerator = torch.einsum('t,tc,to->co', weights, 
                                   predict_label[mask], 
                                   self.dataset_tensor[mask,w])
            denominator = torch.einsum('t,tc->c', weights, predict_label[mask]) + epsilon
            base_confusion[w] = numerator / denominator.unsqueeze(1)
            
            # Apply L2 regularization toward population average
            pop_avg = numerator.sum(dim=0) / denominator.sum() + epsilon
            base_confusion[w] = (base_confusion[w] + self.l2_reg * pop_avg) / (1 + self.l2_reg)
            
        self.base_confusion = base_confusion
        
        # Update calibration parameters via gradient descent
        optimizer = torch.optim.Adam([self.calib_slope, self.calib_bias], lr=0.1)
        optimizer.zero_grad()
        
        # Loss: Encourage high confidence to correlate with accuracy
        diag_vals = torch.stack([self._get_confusion_matrix(w, 1.0).diag().mean() 
                               for w in range(self.worker_num)])
        loss = -torch.log(diag_vals + 1e-10).mean()
        loss.backward()
        optimizer.step()
        
        return base_confusion

    def _e_step(self, predict_label, error_rates):
        log_posterior = torch.zeros_like(predict_label)
        
        for t in range(self.task_num):
            workers = torch.where(self.dataset_tensor[t].sum(dim=1) > 0)[0]
            confidences = self.dataset_tensor[t,workers].max(dim=1)[0]
            
            log_likelihood = 0
            for w, c in zip(workers, confidences):
                cm = self._get_confusion_matrix(w, c)
                log_likelihood += torch.log(cm + 1e-10) @ self.dataset_tensor[t,w]
                
            log_posterior[t] = torch.log(predict_label[t] + 1e-10) + log_likelihood
            log_posterior[t] -= torch.logsumexp(log_posterior[t], dim=0)
            
        return torch.exp(log_posterior)

    def get_reliability(self):
        """Continuous reliability metric considering confidence calibration"""
        # Effective accuracy at maximum confidence
        max_conf_acc = torch.stack([self._get_confusion_matrix(w, 1.0).diag().mean()
                                  for w in range(self.worker_num)])
        
        # Calibration quality (slope of accuracy vs. confidence)
        conf_scores = torch.linspace(0, 1, 100)
        accs = torch.stack([self._get_confusion_matrix(w, c).diag().mean()
                          for w in range(self.worker_num) for c in conf_scores])
        slopes = (accs[:, -10:].mean(dim=1) - accs[:, :10].mean(dim=1))
        
        return max_conf_acc * torch.exp(slopes)

import torch
import torch.nn.functional as F
import logging

class QuantileBinnedDawidSkene:
    def __init__(self,
                 class_num,
                 max_iter=100,
                 tolerance=0.01,
                 batch_size=1000) -> None:
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size  # Adjust if memory is limited

    def run(self, dataset):
        """
        dataset: A FloatTensor of shape (T, W, C)
            - T = number of tasks
            - W = number of workers
            - C = number of classes
          Each dataset[t, w, :] is the post-softmax probabilities from worker w on task t.
        
        We'll build:
          confidence_bin[t, w] ∈ {0,1,2}
             (depending on each worker’s own quartiles of max probability)
        Then hold 3 confusion matrices for each worker:
          error_rates[w, bin, actual_class, reported_class]
        """
        device = dataset.device
        self.device = device
        self.task_num, self.worker_num, _ = dataset.shape
        self.dataset_tensor = dataset
        T, W, C = dataset.shape

        # ---------------------------------------------------------------------
        # Step 1. Assign each (task, worker) pair to a confidence bin {0,1,2},
        #         using worker-specific quartiles of their max probability.
        # ---------------------------------------------------------------------
        max_scores, _ = self.dataset_tensor.max(dim=2)  # shape [T, W]
        confidence_bin = torch.full_like(max_scores, fill_value=-1, dtype=torch.long)

        for w in range(W):
            worker_scores = max_scores[:, w]
            q1 = torch.quantile(worker_scores, 0.33)
            q3 = torch.quantile(worker_scores, 0.67)

            confidence_bin[worker_scores < q1, w] = 0
            confidence_bin[(worker_scores >= q1) & (worker_scores < q3), w] = 1
            confidence_bin[worker_scores >= q3, w] = 2

        # We'll keep this around so both M-step and E-step can see it
        self.confidence_bin = confidence_bin

        # ---------------------------------------------------------------------
        # Initialize predict_label:
        #   predict_label[t] is a vector of length C that represents the
        #   "soft" distribution over the true class of task t.
        #
        # A common choice is to start from the average across workers.
        # ---------------------------------------------------------------------
        mean_over_workers = self.dataset_tensor.mean(dim=1)  # [T, C]
        predict_label = mean_over_workers / (mean_over_workers.sum(dim=1, keepdim=True) + 1e-10)

        # EM loop:
        flag = True
        prev_error_rates, prev_predict_label = None, None
        iter_num = 0

        while flag:
            # M-step: update confusion matrices
            error_rates = self._m_step(predict_label)

            # E-step: update predict_label
            next_predict_label = self._e_step(predict_label, error_rates)

            # Evaluate log-likelihood for debugging
            log_L = self._get_likelihood(predict_label, error_rates)

            if iter_num == 0:
                logging.info("{}\t{}".format(iter_num, log_L))
            else:
                # Check convergence
                marginal_predict = torch.sum(predict_label, 0) / self.task_num
                prev_marginal_predict = torch.sum(prev_predict_label, 0) / self.task_num
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

                print(f"Iteration {iter_num}: marginals_diff={marginals_diff.item():.5f} "
                      f"error_rates_diff={error_rates_diff.item():.5f} log_L={log_L:.5f}")

            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iter_num += 1

        # Once converged, compute final marginal label distribution:
        marginal_predict = torch.sum(predict_label, 0) / self.task_num

        # Compute "worker reliability" as the sum of diagonal for each of the 3 confusion matrices
        # weighted by how often each bin is used.
        # --------------------------------------------------------------------
        # 1) Count how many tasks fall into each bin for each worker: frac_uses[w,b]
        # --------------------------------------------------------------------
        bin_one_hot = torch.nn.functional.one_hot(confidence_bin, num_classes=3).float() 
        bin_usage_counts = bin_one_hot.sum(dim=0)  # shape [W,3]
        bin_usage_fraction = bin_usage_counts / float(T)

        # --------------------------------------------------------------------
        # 2) Probability of correctness per bin: 
        #    sum_i( marginal_predict[i] * error_rates[w,b,i,i] )
        # --------------------------------------------------------------------
        diag_ = torch.diagonal(error_rates, offset=0, dim1=2, dim2=3)  # shape [W,3,C]
        prob_correct = torch.einsum('wbc,c->wb', diag_, marginal_predict)

        # --------------------------------------------------------------------
        # 3) Weighted average: reliability[w] = ∑_{b} bin_usage_fraction[w,b] * prob_correct[w,b]
        # --------------------------------------------------------------------
        reliability = torch.einsum('wb,wb->w', bin_usage_fraction, prob_correct)

        return marginal_predict, error_rates, reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return ((marginals_diff < self.tolerance and error_rates_diff < self.tolerance) 
                or iter_num > self.max_iter)

    # -------------------------------------------------------------------------
    # M-step: Recompute "error_rates" = [W, 3, C, C].
    # -------------------------------------------------------------------------
    def _m_step(self, predict_label):
        device = predict_label.device
        epsilon = 1e-10

        # We'll store 3 confusion matrices per worker: shape [W, 3, C, C]
        error_rates = torch.zeros((self.worker_num, 3, self.class_num, self.class_num),
                                  dtype=torch.float32, device=device)

        # For memory reasons, do chunks of workers:
        for start in range(0, self.worker_num, self.batch_size):
            end = min(start + self.batch_size, self.worker_num)

            # shape: [T, (end-start), C]
            batch_dataset = self.dataset_tensor[:, start:end, :]
            # shape: [T, (end-start)]
            batch_conf_bin = self.confidence_bin[:, start:end]

            # 1) One-hot version of the confidence bin: shape => [T, (end-start), 3]
            bin_onehot = F.one_hot(batch_conf_bin, num_classes=3).float()

            # 2) Compute partial confusion counts via einsum:
            worker_error_rate = torch.einsum(
                'ti, twc, twb -> wbic',
                predict_label, 
                batch_dataset, 
                bin_onehot
            )
            # worker_error_rate has shape [batch_size, 3, C, C]

            # 3) Normalize across reported_class dimension
            sum_over_reported = worker_error_rate.sum(dim=3, keepdim=True) + epsilon
            worker_error_rate = worker_error_rate / sum_over_reported

            # Store
            error_rates[start:end] = worker_error_rate

        return error_rates

    # -------------------------------------------------------------------------
    # E-step:
    # -------------------------------------------------------------------------
    def _e_step(self, predict_label, error_rates):
        device = predict_label.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        marginal_probability = torch.sum(predict_label, dim=0) / T  # shape [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)  # shape [C]

        # We'll use log_pi = log( error_rates ), shape [W,3,C,C]
        log_pi = torch.log(error_rates + 1e-10)

        next_predict_label = torch.zeros([T, C], dtype=torch.float32, device=device)

        # Process tasks in batches so we don't blow up memory
        batch_size = self.batch_size
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)

            batch_dataset = self.dataset_tensor[start:end]  
            batch_conf_bin = self.confidence_bin[start:end]

            # shape [T', C]
            log_class_likelihood = torch.zeros((end - start, C), device=device)

            for w in range(W):
                b_idx = batch_conf_bin[:, w]  # each entry ∈ {0,1,2}
                worker_log_pi = log_pi[w]     # shape [3, C, C]
                # shape [T', 3, C]
                temp = torch.einsum('tc,bkc->tbk', batch_dataset[:, w], worker_log_pi)
                # shape [T', C]
                chosen_bins = temp[range(end - start), b_idx, :]
                log_class_likelihood += chosen_bins

            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)

            next_predict_label[start:end] = torch.exp(log_posterior)

        return next_predict_label

    # -------------------------------------------------------------------------
    # Compute likelihood (in log form) for debugging. Similar logic as E-step:
    # -------------------------------------------------------------------------
    def _get_likelihood(self, predict_label, error_rates):
        device = predict_label.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        marginal_probability = torch.sum(predict_label, dim=0) / T  # shape [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)
        log_pi = torch.log(error_rates + 1e-10)

        log_L = 0.0
        batch_size = self.batch_size
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_dataset = self.dataset_tensor[start:end]  
            batch_conf_bin = self.confidence_bin[start:end]

            log_class_likelihood = torch.zeros((end - start, C), device=device)

            for w in range(W):
                b_idx = batch_conf_bin[:, w]
                worker_log_pi = log_pi[w]  # [3, C, C]
                temp = torch.einsum('tc,bkc->tbk', batch_dataset[:, w], worker_log_pi)
                chosen_bins = temp[range(end - start), b_idx, :]
                log_class_likelihood += chosen_bins

            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_L += torch.sum(torch.logsumexp(log_posterior, dim=1))

        return log_L.item()

import torch
import torch.nn.functional as F
import logging

class ConfidenceWeightedDawidSkene:
    def __init__(self,
                 class_num,
                 max_iter=100,
                 tolerance=0.01,
                 batch_size=1000) -> None:
        """
        class_num: number of classes
        max_iter: maximum number of EM iterations
        tolerance: convergence threshold
        batch_size: to control GPU/CPU memory usage when summing across workers/tasks
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size  # We keep chunked processing for memory reasons

    def run(self, dataset, confidence_scores):
        """
        Args:
            dataset: A FloatTensor of shape (T, W, C)
                - T = number of tasks
                - W = number of workers
                - C = number of classes
              Each dataset[t, w, :] is the post-softmax probabilities from worker w on task t.
            
            confidence_scores: A FloatTensor of shape (T, W)
                - confidence_scores[t, w] is a non-negative number indicating
                  how "confident" worker w is about task t. If you only have partial
                  info, you can pass all 1.0s or some heuristic measure.
        Returns:
            marginal_predict: [C], the final global distribution over classes
            error_rates: [W, C, C], each worker’s confusion matrix
            reliability: [W], overall measure of each worker’s reliability
            predict_label: [T, C], the posterior distribution for each task
        """

        device = dataset.device
        self.device = device
        T, W, C = dataset.shape
        self.task_num = T
        self.worker_num = W
        self.dataset_tensor = dataset  # shape [T, W, C]

        # Store confidence scores for use in M-step/E-step
        self.confidence_scores = confidence_scores.to(device)

        # ---------------------------------------------------------------------
        # Step 0: Initialize the distribution over the true labels of each task.
        #         We'll take the average of the workers' softmax predictions,
        #         then normalize. [T, C]
        # ---------------------------------------------------------------------
        mean_over_workers = self.dataset_tensor.mean(dim=1)  # shape [T, C]
        predict_label = mean_over_workers / (mean_over_workers.sum(dim=1, keepdim=True) + 1e-10)

        # We'll iteratively refine:
        flag = True
        prev_error_rates, prev_predict_label = None, None
        iter_num = 0

        while flag:
            # M-step: update confusion matrices
            error_rates = self._m_step(predict_label)

            # E-step: update predict_label
            next_predict_label = self._e_step(predict_label, error_rates)

            # Evaluate log-likelihood for debugging
            log_L = self._get_likelihood(predict_label, error_rates)

            if iter_num == 0:
                logging.info("{}\t{}".format(iter_num, log_L))
            else:
                # Check convergence
                marginal_predict = torch.sum(predict_label, 0) / T
                prev_marginal_predict = torch.sum(prev_predict_label, 0) / T
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

                print(f"Iteration {iter_num}: marginals_diff={marginals_diff.item():.5f} "
                      f"error_rates_diff={error_rates_diff.item():.5f} log_L={log_L:.5f}")

            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iter_num += 1

        # Once converged, compute final marginal label distribution:
        marginal_predict = torch.sum(predict_label, 0) / T

        # --------------------------------------------------------------------
        # "Worker Reliability" as the sum of diagonal entries in each confusion matrix
        # We'll just measure the average correctness under the final global class distribution.
        # --------------------------------------------------------------------
        # Diagonal: error_rates[w, k, k]
        diag_ = torch.diagonal(error_rates, offset=0, dim1=1, dim2=2)  # shape [W, C]
        # Weighted by the final marginal_predict over classes => [C]
        prob_correct = torch.einsum('wc,c->w', diag_, marginal_predict)
        # We’ll call that reliability
        reliability = prob_correct

        return marginal_predict, error_rates, reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return ((marginals_diff < self.tolerance and error_rates_diff < self.tolerance) 
                or iter_num > self.max_iter)

    # -------------------------------------------------------------------------
    # M-step: Recompute "error_rates" = [W, C, C]
    #         Weighted by confidence_scores[t,w].
    #
    #   error_rates[w, i, c] = Probability(worker w reports class c | true class i).
    #
    #   We approximate by:
    #     sum_{t} [ s_{t,w} * predict_label[t, i] * dataset[t, w, c] ] 
    #     / sum_{t} [ s_{t,w} * predict_label[t, i] ].
    #
    #  Doing it in chunked fashion for memory reasons.
    # -------------------------------------------------------------------------
    def _m_step(self, predict_label):
        device = predict_label.device
        epsilon = 1e-10

        error_rates = torch.zeros((self.worker_num, self.class_num, self.class_num),
                                  dtype=torch.float32, device=device)

        for start in range(0, self.worker_num, self.batch_size):
            end = min(start + self.batch_size, self.worker_num)

            # slice: shape [T, batch_size, C]
            batch_dataset = self.dataset_tensor[:, start:end, :]
            # shape [T, batch_size]
            batch_conf = self.confidence_scores[:, start:end]

            # We want partial sums for each worker in this slice:
            # 
            # Numerator for each w_in_batch, i, c:
            #   sum_{t} s_{t,w} * predict_label[t, i] * dataset[t, w, c]
            #
            # We'll use an einsum trick:
            #   'ti, twc, tw -> wic'
            #    t=task, i=true_class, w=worker_in_batch, c=reported_class
            #
            # Then we sum over t.
            worker_counts = torch.einsum(
                'ti, twc, tw -> wic',
                predict_label, 
                batch_dataset, 
                batch_conf
            )
            # shape [batch_size, C, C]

            # Denominator for each w_in_batch, i:
            #   sum_{t} s_{t,w} * predict_label[t, i]
            #
            #   'ti, tw -> wi'
            denom = torch.einsum(
                'ti, tw -> wi',
                predict_label,
                batch_conf
            )
            # shape [batch_size, C]

            # Normalize:
            denom = denom.unsqueeze(-1) + epsilon  # shape [batch_size, C, 1]
            worker_error_rate = worker_counts / denom  # shape [batch_size, C, C]

            # Store into the full array
            error_rates[start:end] = worker_error_rate

        return error_rates

    # -------------------------------------------------------------------------
    # E-step:
    #   next_predict_label[t, k] ∝ prior[k] * ∏_{w} ( ∑_{c} [ s_{t,w}-weighted factor ] )
    #
    # But we do it in log-space for stability. For each (t, w), we incorporate
    #   log( ∑_{c} dataset[t,w,c] * error_rates[w, k, c] ) * confidence_scores[t,w]
    #
    # i.e., we multiply the log-likelihood contribution by confidence, or equivalently,
    # scale it by s_{t,w} inside the log. That’s the trick. 
    #
    # We still chunk over tasks in batches to keep memory usage contained.
    # -------------------------------------------------------------------------
    def _e_step(self, predict_label, error_rates):
        device = predict_label.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        # prior: a simple "empirical prior" from the current predict_label
        marginal_probability = torch.sum(predict_label, dim=0) / T  # shape [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        # We'll accumulate new predictions in next_predict_label
        next_predict_label = torch.zeros([T, C], dtype=torch.float32, device=device)

        batch_size = self.batch_size
        log_error_rates = torch.log(error_rates + 1e-10)  # shape [W, C, C]

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)

            # shape [T', W, C]
            batch_dataset = self.dataset_tensor[start:end]  
            # shape [T', W]
            batch_conf = self.confidence_scores[start:end]

            Tprime = end - start
            # We'll compute log_likelihood[t', k] for each t' in [0, T') and k in [0, C].
            log_class_likelihood = torch.zeros((Tprime, C), device=device)

            # Accumulate over each worker
            for w in range(W):
                # shape [C, C]
                worker_log_err = log_error_rates[w]
                # shape [T', C]
                # For each (t', c_reported), we want dataset[t', w, c_reported] * error_rates[w, k, c_reported]
                # We'll do an einsum to sum over c_reported. Then we take the log of that sum (safe log-sum-exp).
                #
                # But we want to weigh by s_{t', w} in the exponent => multiply the log-likelihood by s_{t', w}.
                # That means we do something like:
                #   "weighted_log_factor = s_{t', w} * log( sum_c( dataset * exp(worker_log_err[k,c]) ) )"
                # We'll do it carefully in log space:
                #
                # Let:
                #   sum_over_c[k, t'] = ∑_c [ dataset[t', w, c] * exp(worker_log_err[k, c]) ]
                #
                # Then log(sum_over_c[k, t']) is the log-likelihood contribution for worker w if true class = k,
                # ignoring the weighting for the moment. Then we multiply by s_{t', w}.
                #
                # We'll do a more direct approach with log-sum-exp:
                #   For each k, t':
                #     log( sum_c( dataset[t', w, c]*exp(worker_log_err[k,c]) ) )
                #     = logsumexp_c( log(dataset[t', w, c]) + worker_log_err[k,c] )
                #
                # Then multiply the result by batch_conf[t', w].
                #
                # We'll do a partial vectorization:
                #   1) shape [C, T']: each row is log( dataset[t', w, c ] )
                #   2) we add worker_log_err[k,c] (for each k) => shape [k, c, T']
                #   3) logsumexp over c => shape [k, T']
                #   4) multiply by s_{t', w}.
                #   5) transpose to [T', k]
                #
                # We'll accumulate that in log_class_likelihood[t', k].
                #
                batch_log_dataset = torch.log(batch_dataset[:, w, :] + 1e-10)  # [T', C]
                # We want to broadcast over k:
                #   worker_log_err -> shape [C, C]
                #   We'll do an expand to [C, 1, C] or something. Then combine with batch_log_dataset [T', C].
                # Easiest is an outer loop, but let's do a more direct approach:
                # shape [C, T', C]
                repeated_log_err = worker_log_err.unsqueeze(1).expand(C, Tprime, C)
                repeated_log_data = batch_log_dataset.unsqueeze(0).expand(C, Tprime, C)
                # shape [C, T', C] => sum them
                sum_ = repeated_log_err + repeated_log_data
                # Now logsumexp over c dimension => shape [C, T']
                partial = torch.logsumexp(sum_, dim=2)
                # Multiply by confidence score s_{t', w}
                # shape [C, T'] => [T', C] after transpose
                weighted_contrib = (partial * batch_conf[:, w].unsqueeze(0)).transpose(0,1)
                # Accumulate
                log_class_likelihood += weighted_contrib

            # Now add the prior
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            # Normalize over k
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)

            next_predict_label[start:end] = torch.exp(log_posterior)

        return next_predict_label

    # -------------------------------------------------------------------------
    # Compute Weighted Log-Likelihood, to monitor progress in each iteration.
    #   log_L = Σ_t log( Σ_k [ prior[k] * ∏_w WeightedTerm ] )
    # with WeightedTerm = [ sum_c dataset[t,w,c] * error_rates[w,k,c] ]^( confidence_scores[t,w] ).
    # We do it in mini-batches for memory.
    # -------------------------------------------------------------------------
    def _get_likelihood(self, predict_label, error_rates):
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        marginal_probability = torch.sum(predict_label, dim=0) / T  # [C]
        log_marginal_probability = torch.log(marginal_probability + 1e-10)
        log_error_rates = torch.log(error_rates + 1e-10)

        log_L = 0.0
        batch_size = self.batch_size

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_dataset = self.dataset_tensor[start:end]  
            batch_conf = self.confidence_scores[start:end]
            Tprime = end - start

            # shape [Tprime, C]
            log_class_likelihood = torch.zeros((Tprime, C), device=batch_dataset.device)

            for w in range(W):
                worker_log_err = log_error_rates[w]  # shape [C, C]

                # We'll do the same summation and log-sum-exp as in E-step:
                batch_log_dataset = torch.log(batch_dataset[:, w, :] + 1e-10)  # [Tprime, C]
                # shape [C, Tprime, C]
                repeated_log_err = worker_log_err.unsqueeze(1).expand(C, Tprime, C)
                repeated_log_data = batch_log_dataset.unsqueeze(0).expand(C, Tprime, C)
                sum_ = repeated_log_err + repeated_log_data
                partial = torch.logsumexp(sum_, dim=2)  # shape [C, Tprime]

                # Weighted by confidence
                weighted_contrib = (partial * batch_conf[:, w].unsqueeze(0)).transpose(0,1)
                log_class_likelihood += weighted_contrib

            # Combine with prior
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            # sum over tasks of log-sum-exp
            log_L += torch.sum(torch.logsumexp(log_posterior, dim=1))

        return log_L.item()

import torch
import torch.nn.functional as F
import logging

class SkillConfidenceWeightedDawidSkene:
    def __init__(self,
                 class_num,
                 max_iter=100,
                 tolerance=0.01,
                 batch_size=1000,
                 skill_lr=0.01) -> None:
        """
        Args:
            class_num: number of classes
            max_iter: maximum number of EM iterations
            tolerance: convergence threshold for stopping
            batch_size: chunk size for GPU memory optimization
            skill_lr: learning rate for updating worker skill parameters alpha_w
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.skill_lr = skill_lr  # step size for skill gradient updates

    def run(self, dataset, confidence_scores):
        """
        Args:
            dataset: A FloatTensor of shape (T, W, C)
                - T = number of tasks
                - W = number of workers
                - C = number of classes
              dataset[t, w, c] = probability that worker w says class c on task t.
            
            confidence_scores: FloatTensor of shape (T, W)
                - s_{t,w} is the confidence for worker w on task t.
        
        Returns:
            marginal_predict: [C], the final global distribution over classes
            error_rates: [W, C, C], each worker’s confusion matrix
            reliability: [W], a final “reliability” measure for each worker 
                         (in this code, we’ll simply return alpha_w).
            predict_label: [T, C], posterior distribution for each task
        """
        device = dataset.device
        self.device = device
        T, W, C = dataset.shape
        self.task_num = T
        self.worker_num = W
        self.dataset_tensor = dataset  # shape [T, W, C]
        self.confidence_scores = confidence_scores.to(device)

        # Initialize worker skill alpha: shape [W], start at 1.0
        self.worker_skill = torch.ones(W, device=device, requires_grad=False)

        # Initialize predict_label from average over workers
        mean_over_workers = self.dataset_tensor.mean(dim=1)  # shape [T, C]
        predict_label = mean_over_workers / (mean_over_workers.sum(dim=1, keepdim=True) + 1e-10)

        # We'll also keep confusion matrices. We'll update them in the M-step. 
        # Start them roughly uniform or something stable:
        error_rates = torch.ones((W, C, C), dtype=torch.float32, device=device)
        error_rates = error_rates / error_rates.sum(dim=2, keepdim=True)

        flag = True
        prev_error_rates = error_rates.clone()
        prev_predict_label = predict_label.clone()
        iter_num = 0

        while flag:
            # M-step: update confusion matrices using the current skill + confidence weighting
            error_rates = self._m_step(predict_label)

            # E-step: update predict_label with skill weighting
            next_predict_label = self._e_step(predict_label, error_rates)

            # "Skill-step": update alpha_w to maximize log-likelihood
            self._update_worker_skill(next_predict_label, error_rates)

            # Evaluate log-likelihood for debugging
            log_L = self._get_likelihood(next_predict_label, error_rates)

            # Check for convergence
            if iter_num == 0:
                logging.info("{}\t{}".format(iter_num, log_L))
            else:
                # compute diffs
                marginal_predict = torch.sum(predict_label, 0) / T
                prev_marginal_predict = torch.sum(prev_predict_label, 0) / T
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

                print(f"Iteration {iter_num}: marginals_diff={marginals_diff.item():.5f} "
                      f"error_rates_diff={error_rates_diff.item():.5f} log_L={log_L:.5f}")

            prev_error_rates = error_rates.clone()
            prev_predict_label = predict_label.clone()
            predict_label = next_predict_label
            iter_num += 1

        # Final marginal distribution over classes
        marginal_predict = torch.sum(predict_label, 0) / T

        # We’ll define the reliability as the skill alpha_w for now
        # (You could also incorporate the confusion matrix’s diagonal if you want.)
        reliability = self.worker_skill.detach().clone()

        return marginal_predict, error_rates, reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return ((marginals_diff < self.tolerance and error_rates_diff < self.tolerance) 
                or iter_num >= self.max_iter)

    # -------------------------------------------------------------------------
    # M-step: Weighted by alpha_w * confidence_scores[t,w].
    # error_rates[w, i, c] \propto sum_{t} alpha_w * s_{t,w} * predict_label[t,i] * dataset[t,w,c].
    # Then normalize across c.
    # -------------------------------------------------------------------------
    def _m_step(self, predict_label):
        device = self.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        epsilon = 1e-10
        error_rates = torch.zeros((W, C, C), dtype=torch.float32, device=device)

        alpha = self.worker_skill  # shape [W]

        for start in range(0, W, self.batch_size):
            end = min(start + self.batch_size, W)

            batch_dataset = self.dataset_tensor[:, start:end, :]   # [T, batch_size, C]
            batch_conf = self.confidence_scores[:, start:end]      # [T, batch_size]
            batch_alpha = alpha[start:end]                         # [batch_size]

            # We'll do:
            #   worker_counts[w_in_batch, i, c] = sum_t ( alpha[w_in_batch] * s_{t,w} 
            #                                             * predict_label[t,i] 
            #                                             * dataset[t,w,c] )
            # shape => [batch_size, C, C]
            worker_counts = torch.einsum(
                'ti, twc, tw, w -> wic',
                predict_label, 
                batch_dataset, 
                batch_conf,
                batch_alpha
            )
            # Summation is over t

            # Denominator: sum_c(...) for normalization. We'll do:
            #   denom[w_in_batch, i] = sum_c( worker_counts[w_in_batch, i, c] )
            # shape => [batch_size, C]
            denom = worker_counts.sum(dim=2, keepdim=True) + epsilon
            worker_error_rate = worker_counts / denom

            error_rates[start:end] = worker_error_rate

        return error_rates

    # -------------------------------------------------------------------------
    # E-step: next_predict_label[t,k] ∝ prior[k] * ∏_w( sum_c dataset[t,w,c]*error_rates[w,k,c] )^( alpha_w * s_{t,w} )
    #
    # We'll do it in log space for stability:
    #   log_class_likelihood[t,k] = sum_w [ alpha_w * s_{t,w} * log( sum_c( dataset[t,w,c]*error_rates[w,k,c] ) ) ]
    # Then exponentiate + normalize over k.
    # -------------------------------------------------------------------------
    def _e_step(self, predict_label, error_rates):
        device = self.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        # prior: from the old predict_label
        marginal_probability = torch.sum(predict_label, dim=0) / T
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        alpha = self.worker_skill  # shape [W]
        log_error_rates = torch.log(error_rates + 1e-10)  # [W, C, C]

        next_predict_label = torch.zeros((T, C), dtype=torch.float32, device=device)

        batch_size = self.batch_size
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            Tprime = end - start

            batch_dataset = self.dataset_tensor[start:end]      # [T', W, C]
            batch_conf = self.confidence_scores[start:end]      # [T', W]

            # shape [T', C]
            log_class_likelihood = torch.zeros((Tprime, C), device=device)

            # accumulate over each worker
            for w in range(W):
                # alpha_w
                aw = alpha[w]
                # shape [C, C]
                w_log_err = log_error_rates[w]
                # shape [T', C]
                # we'll do log( sum_c dataset[t', w, c]*exp(w_log_err[k,c]) ) for each k
                # Then multiply by (aw * s_{t', w})
                # We can do partial vectorization with log-sum-exp:

                batch_log_dataset = torch.log(batch_dataset[:, w, :] + 1e-10)  # [T', C]
                # shape => [C, T', C]
                expanded_log_err = w_log_err.unsqueeze(1).expand(C, Tprime, C)
                expanded_log_data = batch_log_dataset.unsqueeze(0).expand(C, Tprime, C)

                sum_ = expanded_log_err + expanded_log_data
                # shape => [C, T']
                partial = torch.logsumexp(sum_, dim=2)

                # multiply by (aw * s_{t', w}) => shape [T', C] after transpose
                weighted_contrib = (partial * (aw * batch_conf[:, w]).unsqueeze(0)).transpose(0,1)
                log_class_likelihood += weighted_contrib

            # add prior, then normalize
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)

            next_predict_label[start:end] = torch.exp(log_posterior)

        return next_predict_label

    # -------------------------------------------------------------------------
    # Skill-step: update alpha_w to maximize log-likelihood, given the new tasks’ labels posterior.
    # We do a small gradient step for each w, chunked if needed.
    #
    # In log form, derivative wrt alpha_w is:
    #   d/d(alpha_w) sum_{t} [ s_{t,w} * log( sum_c( dataset[t,w,c]*error_rates[w,k,c] ) ) * P(y_t=k) ]
    # for all k weighted by the posterior P(y_t=k).
    #
    # We'll approximate it by summation over tasks t, classes k (from posterior).
    # Then we do alpha_w = alpha_w + skill_lr * grad, clipped to >= 0.
    # -------------------------------------------------------------------------
    def _update_worker_skill(self, predict_label, error_rates):
        device = self.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num
        alpha = self.worker_skill  # shape [W]

        # We'll store the gradient dL/d(alpha_w)
        grad_alpha = torch.zeros_like(alpha, dtype=torch.float32, device=device)

        log_error_rates = torch.log(error_rates + 1e-10)

        # We'll do a loop over w in chunks to avoid huge memory usage
        for start_w in range(0, W, self.batch_size):
            end_w = min(start_w + self.batch_size, W)
            w_size = end_w - start_w

            # shape => [w_size, C, C]
            batch_err = error_rates[start_w:end_w]
            batch_log_err = log_error_rates[start_w:end_w]

            # We'll accumulate partial derivatives for this batch, shape [w_size]
            batch_grad = torch.zeros(w_size, device=device)

            # We'll do tasks in sub-batches too (for large T).
            # The partial derivative for alpha_w is:
            #   sum_{t} s_{t,w} * [ sum_{k} p(t,k) * log( sum_{c} dataset[t,w,c]*error_rates[w,k,c] ) ]
            # We’ll handle everything in log space carefully.
            # We can’t easily factor out alpha because the derivative is w.r.t alpha multiplying that log() term.
            #
            # derivative d/d alpha [ alpha * X ] = X
            # So if L = sum_{t,k} alpha*s_{t,w} * p(t,k) * log( sum_{c} ... ), derivative w.r.t alpha is 
            # sum_{t,k} s_{t,w} * p(t,k) * log( sum_{c} ... ) (no extra chain rule needed except for the sum_{c}... part).
            #
            # We'll do a partial sum over tasks in chunks.

            for start_t in range(0, T, self.batch_size):
                end_t = min(start_t + self.batch_size, T)
                t_size = end_t - start_t

                # shape => [t_size, w_size, C]
                sub_dataset = self.dataset_tensor[start_t:end_t, start_w:end_w, :]
                sub_conf = self.confidence_scores[start_t:end_t, start_w:end_w]  # [t_size, w_size]
                sub_predict = predict_label[start_t:end_t]                      # [t_size, C]

                # We want:
                #   for each w_in_batch, sum over t, sum over k [ s_{t,w} * p(t,k) * log( sum_c( dataset[t,w,c]*exp( batch_log_err[w_in_batch,k,c] ) ) ) ]
                # We'll do partial vectorization:

                # 1) shape => [t_size, w_size, 1, C]
                sub_logdata = torch.log(sub_dataset + 1e-10).unsqueeze(2)  # we want to eventually broadcast over k

                # 2) shape => [w_size, C, C]
                # We'll need a trick. Probably easiest is a double loop over w_in_batch, but we can do an einsum for the sum over c.  
                # However, we still must incorporate the "logsumexp" over c for each k. We'll do it for each w_in_batch individually.
                # 
                # We'll do it in a loop to keep it straightforward.

                for wb in range(w_size):
                    w_idx = start_w + wb
                    # shape [C, C]
                    w_log_err = batch_log_err[wb]
                    # sub_conf[:, wb], shape [t_size]
                    conf_w = sub_conf[:, wb]

                    # We'll accumulate partial derivative for this w_in_batch
                    partial_grad = 0.0

                    # We'll do tasks in a chunk: shape => [t_size, C]
                    # log( sum_c( dataset[t,w_idx,c] * exp( w_log_err[k,c] ) ) )
                    # Then multiply that by s_{t,w_idx} * sum_k p(t,k).
                    # 
                    # But we actually need sum_k [ p(t,k)* that log(...) ] if the derivative is w.r.t alpha. 
                    # Let's define:
                    #   log_term[t, k] = log( sum_c( dataset[t,w_idx,c] * exp( w_log_err[k,c] ) ) )
                    # Then sum over k => sum_k [ p(t,k)* log_term[t,k] ] 
                    # Then multiply by conf_w[t].
                    #
                    # We'll do a partial vectorization:
                    dataset_t_w = sub_dataset[:, wb, :]  # shape [t_size, C]
                    log_dataset_t_w = torch.log(dataset_t_w + 1e-10)  # [t_size, C]
                    # shape => [C, t_size, C]
                    repeated_w_log_err = w_log_err.unsqueeze(1).expand(C, t_size, C)
                    repeated_log_data = log_dataset_t_w.unsqueeze(0).expand(C, t_size, C)
                    sum_ = repeated_w_log_err + repeated_log_data
                    # shape => [C, t_size]
                    logsumexp_c = torch.logsumexp(sum_, dim=2)

                    # shape => [t_size, C], transpose
                    logsumexp_c_tk = logsumexp_c.transpose(0,1)

                    # Weighted by p(t,k)
                    # shape => [t_size, C]
                    # multiply by p(t,k), then sum over k => shape [t_size]
                    weighted_by_p = logsumexp_c_tk * sub_predict[:, :]
                    sum_over_k = weighted_by_p.sum(dim=1)

                    # Finally multiply by s_{t,w_idx}
                    partial_grad += torch.sum( conf_w * sum_over_k )

                    batch_grad[wb] += partial_grad

            # Now batch_grad[wb] is the partial derivative for alpha_{start_w+wb}.
            # We'll do a simple gradient step:
            updated_alpha = alpha[start_w:end_w] + self.skill_lr * batch_grad
            updated_alpha = torch.clamp(updated_alpha, min=0.0)
            alpha[start_w:end_w] = updated_alpha

        # Store back
        self.worker_skill = alpha

    # -------------------------------------------------------------------------
    # _get_likelihood: Weighted + skill-scaled log-likelihood
    # We do this for monitoring. 
    # -------------------------------------------------------------------------
    def _get_likelihood(self, predict_label, error_rates):
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        alpha = self.worker_skill
        log_error_rates = torch.log(error_rates + 1e-10)

        # prior
        marginal_probability = torch.sum(predict_label, dim=0)/T
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        log_L = 0.0
        batch_size = self.batch_size
        for start_t in range(0, T, batch_size):
            end_t = min(start_t + batch_size, T)
            t_size = end_t - start_t
            batch_dataset = self.dataset_tensor[start_t:end_t]  # [t_size, W, C]
            batch_conf = self.confidence_scores[start_t:end_t]  # [t_size, W]
            batch_predict = predict_label[start_t:end_t]        # [t_size, C]

            # shape => [t_size, C]
            log_class_likelihood = torch.zeros((t_size, C), device=batch_dataset.device)
            
            for w in range(W):
                aw = alpha[w]
                w_log_err = log_error_rates[w]
                # shape => [t_size, C]
                batch_log_dataset = torch.log(batch_dataset[:, w, :] + 1e-10)

                # shape => [C, t_size, C]
                repeated_log_err = w_log_err.unsqueeze(1).expand(C, t_size, C)
                repeated_log_data = batch_log_dataset.unsqueeze(0).expand(C, t_size, C)

                sum_ = repeated_log_err + repeated_log_data
                # shape => [C, t_size]
                partial = torch.logsumexp(sum_, dim=2)

                # Multiply by (aw * batch_conf[:, w]) => shape => [t_size, C]
                weighted = (partial * (aw * batch_conf[:, w]).unsqueeze(0)).transpose(0,1)
                log_class_likelihood += weighted

            # Add prior
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            # sum over tasks of logsumexp
            log_L += torch.sum(torch.logsumexp(log_posterior, dim=1))

        return log_L.item()

import torch
import torch.nn.functional as F
import logging

class SkillDifficultyConfidenceWeightedDawidSkene:
    def __init__(self,
                 class_num,
                 max_iter=100,
                 tolerance=0.01,
                 batch_size=1000,
                 skill_lr=0.00001) -> None:
        """
        Args:
            class_num: number of classes
            max_iter: maximum number of EM iterations
            tolerance: convergence threshold for stopping
            batch_size: chunk size for GPU memory optimization
            skill_lr: learning rate for updating both worker skill (alpha_w) 
                      and task difficulty (beta_t)
        """
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.skill_lr = skill_lr  # step size for gradient updates on alpha_w & beta_t

    def run(self, dataset, confidence_scores):
        """
        Args:
            dataset: A FloatTensor of shape (T, W, C)
                - T = number of tasks
                - W = number of workers
                - C = number of classes
              dataset[t, w, c] = probability that worker w says class c on task t.
            
            confidence_scores: FloatTensor of shape (T, W)
                - s_{t,w} is the confidence for worker w on task t.
        
        Returns:
            marginal_predict: [C], final global distribution over classes
            error_rates: [W, C, C], each worker’s confusion matrix
            reliability: [W], a final “reliability” measure (alpha_w) for each worker 
            predict_label: [T, C], posterior distribution for each task
            task_difficulty: [T], the learned difficulty parameter for each task
        """
        device = dataset.device
        self.device = device
        T, W, C = dataset.shape
        self.task_num = T
        self.worker_num = W
        self.dataset_tensor = dataset  # shape [T, W, C]
        self.confidence_scores = confidence_scores.to(device)

        # -----------------------------
        # 1) Initialize parameters
        # -----------------------------
        # (a) Worker skill alpha_w: shape [W]
        self.worker_skill = torch.ones(W, device=device, requires_grad=False)
        # (b) Task difficulty beta_t: shape [T]
        self.task_difficulty = torch.zeros(T, device=device, requires_grad=False)
        #     (Starting with 0.0 so initially alpha_w * s_{t,w} is unpenalized by difficulty.)

        # (c) Initialize per-task label distribution from average
        mean_over_workers = self.dataset_tensor.mean(dim=1)  # shape [T, C]
        predict_label = mean_over_workers / (mean_over_workers.sum(dim=1, keepdim=True) + 1e-10)

        # (d) Initialize confusion matrices to something stable (e.g. uniform)
        error_rates = torch.ones((W, C, C), dtype=torch.float32, device=device)
        error_rates = error_rates / error_rates.sum(dim=2, keepdim=True)

        # ------------------------------------
        # 2) EM loop (+ skill & difficulty updates)
        # ------------------------------------
        flag = True
        prev_error_rates = error_rates.clone()
        prev_predict_label = predict_label.clone()
        iter_num = 0

        while flag:
            # M-step: update confusion matrices
            error_rates = self._m_step(predict_label)

            # E-step: update predict_label (log space, then normalize)
            next_predict_label = self._e_step(predict_label, error_rates)

            # Update worker skill alpha_w
            self._update_worker_skill(next_predict_label, error_rates)

            # Update task difficulty beta_t
            self._update_task_difficulty(next_predict_label, error_rates)

            # Evaluate log-likelihood for debugging
            log_L = self._get_likelihood(next_predict_label, error_rates)

            # Check for convergence
            if iter_num == 0:
                logging.info("{}\t{}".format(iter_num, log_L))
            else:
                # compute diffs
                marginal_predict = torch.sum(predict_label, 0) / T
                prev_marginal_predict = torch.sum(prev_predict_label, 0) / T
                marginals_diff = torch.sum(torch.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = torch.sum(torch.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

                print(f"Iteration {iter_num}: marginals_diff={marginals_diff.item():.5f} "
                      f"error_rates_diff={error_rates_diff.item():.5f} log_L={log_L:.5f}")

            prev_error_rates = error_rates.clone()
            prev_predict_label = predict_label.clone()
            predict_label = next_predict_label
            iter_num += 1

        # Final marginal distribution over classes
        marginal_predict = torch.sum(predict_label, 0) / T

        # reliability = alpha_w (the skill)
        reliability = self.worker_skill.detach().clone()

        return marginal_predict, error_rates, reliability, predict_label, self.task_difficulty

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return ((marginals_diff < self.tolerance and error_rates_diff < self.tolerance) 
                or iter_num >= self.max_iter)

    # -------------------------------------------------------------------------
    # M-step (confusion matrices):
    #    error_rates[w, i, c] \propto sum_{t} 
    #         [predict_label[t,i] * dataset[t,w,c] 
    #          * s_{t,w} * (alpha_w - beta_t) ].
    # Then we normalize across c.
    # -------------------------------------------------------------------------
    def _m_step(self, predict_label):
        device = self.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        epsilon = 1e-10
        error_rates = torch.zeros((W, C, C), dtype=torch.float32, device=device)

        alpha = self.worker_skill  # shape [W]
        beta = self.task_difficulty  # shape [T]

        for start in range(0, W, self.batch_size):
            end = min(start + self.batch_size, W)

            batch_dataset = self.dataset_tensor[:, start:end, :]   # [T, batch_size, C]
            batch_conf = self.confidence_scores[:, start:end]      # [T, batch_size]
            batch_alpha = alpha[start:end]                         # [batch_size]

            # We'll accumulate:
            # worker_counts[w_in_batch, i, c] = sum_{t} [ predict_label[t,i] * dataset[t,w,c] 
            #                                  * s_{t,w} * (alpha[w] - beta[t]) ]
            #
            # shape => [batch_size, C, C]
            # worker_counts = torch.einsum(
            #     'ti, twc, tw, t -> wic',
            #     predict_label, 
            #     batch_dataset, 
            #     batch_conf,
            #     (batch_alpha.unsqueeze(0) - beta.unsqueeze(1))  # shape => [T, batch_size]
            # )
            worker_counts = torch.einsum(
                'ti, twc, tw, tw -> wic',
                predict_label, 
                batch_dataset, 
                batch_conf,
                (batch_alpha.unsqueeze(0) - beta.unsqueeze(1))
            )
            # Summation is over t.

            denom = worker_counts.sum(dim=2, keepdim=True) + epsilon
            worker_error_rate = worker_counts / denom

            error_rates[start:end] = worker_error_rate

        return error_rates

    # -------------------------------------------------------------------------
    # E-step:
    #   next_predict_label[t,k] ∝ prior[k] * ∏_w( sum_c dataset[t,w,c]*error_rates[w,k,c] )^((alpha_w - beta_t)* s_{t,w})
    #
    # We'll do it in log space for stability:
    #   log_class_likelihood[t,k] = sum_{w} [ (alpha_w - beta_t)* s_{t,w} 
    #                                         * log( sum_c( dataset[t,w,c]*error_rates[w,k,c] ) ) ]
    # -------------------------------------------------------------------------
    def _e_step(self, predict_label, error_rates):
        device = self.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        # prior from old predict_label
        marginal_probability = torch.sum(predict_label, dim=0) / T
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        alpha = self.worker_skill  # [W]
        beta  = self.task_difficulty  # [T]
        log_error_rates = torch.log(error_rates + 1e-10)  # [W, C, C]

        next_predict_label = torch.zeros((T, C), dtype=torch.float32, device=device)

        batch_size = self.batch_size
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            Tprime = end - start

            batch_dataset = self.dataset_tensor[start:end]        # [Tprime, W, C]
            batch_conf = self.confidence_scores[start:end]        # [Tprime, W]
            # shape => [Tprime]
            batch_beta = beta[start:end]

            # shape => [Tprime, C]
            log_class_likelihood = torch.zeros((Tprime, C), device=device)

            for w in range(W):
                aw = alpha[w]
                w_log_err = log_error_rates[w]  # [C, C]

                # exponent_{t',w} = (aw - batch_beta[t']) * batch_conf[t', w]
                # We'll define a 1D factor for each t': shape [Tprime]
                exponent_vec = (aw - batch_beta) * batch_conf[:, w]

                # Now compute log( sum_c dataset[t',w,c] * exp(w_log_err[k,c]) ) for each t', k
                batch_log_dataset = torch.log(batch_dataset[:, w, :] + 1e-10)  # [Tprime, C]

                # shape => [C, Tprime, C]
                repeated_w_log_err = w_log_err.unsqueeze(1).expand(C, Tprime, C)
                repeated_log_data  = batch_log_dataset.unsqueeze(0).expand(C, Tprime, C)

                sum_ = repeated_w_log_err + repeated_log_data
                # shape => [C, Tprime]
                partial = torch.logsumexp(sum_, dim=2)

                # Multiply by exponent_vec[t']
                # shape => [Tprime, C] after transpose => partial.T => [Tprime, C]
                weighted_contrib = partial.transpose(0,1) * exponent_vec.unsqueeze(1)

                log_class_likelihood += weighted_contrib

            # Now add prior and normalize
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)

            next_predict_label[start:end] = torch.exp(log_posterior)

        return next_predict_label

    # -------------------------------------------------------------------------
    # Skill-step: update alpha_w by gradient. 
    #   exponent_{t,w} = (alpha_w - beta_t)* s_{t,w}.
    #   derivative wrt alpha_w: partial exponent_{t,w} / partial alpha_w = s_{t,w}.
    # -------------------------------------------------------------------------
    def _update_worker_skill(self, predict_label, error_rates):
        device = self.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num
        alpha = self.worker_skill  # shape [W]
        beta  = self.task_difficulty
        log_error_rates = torch.log(error_rates + 1e-10)

        grad_alpha = torch.zeros_like(alpha, dtype=torch.float32, device=device)

        # chunk over w
        for start_w in range(0, W, self.batch_size):
            end_w = min(start_w + self.batch_size, W)

            batch_err = error_rates[start_w:end_w]      # [batch_size, C, C]
            batch_log_err = log_error_rates[start_w:end_w]  # [batch_size, C, C]

            # partial derivatives for alpha_{start_w..end_w}
            batch_grad = torch.zeros(end_w - start_w, device=device)

            # We'll iterate over tasks in sub-batches to avoid giant memory use
            for start_t in range(0, T, self.batch_size):
                end_t = min(start_t + self.batch_size, T)
                t_size = end_t - start_t

                sub_dataset = self.dataset_tensor[start_t:end_t, start_w:end_w, :]  # [t_size, w_size, C]
                sub_conf = self.confidence_scores[start_t:end_t, start_w:end_w]     # [t_size, w_size]
                sub_predict = predict_label[start_t:end_t]                          # [t_size, C]

                # shape => [t_size]
                sub_beta = beta[start_t:end_t]

                # We'll do a loop over each w_in_batch:
                for wb in range(end_w - start_w):
                    w_idx = start_w + wb
                    # derivative wrt alpha_w is sum_{t,k} p(t,k)* 
                    #        [ s_{t,w} * log( sum_c( dataset[t,w,c] * exp(log_err[k,c]) ) ) ]
                    # with exponent_{t,w} = (alpha_w - beta_t)* s_{t,w}, but derivative wrt alpha is s_{t,w}.

                    w_log_err = batch_log_err[wb]
                    # shape => [t_size, C]
                    w_dataset = sub_dataset[:, wb, :]
                    log_w_dataset = torch.log(w_dataset + 1e-10)

                    # We'll accumulate partial derivative
                    partial_grad = 0.0

                    # compute log( sum_c(...) ) for each t in this chunk, for each k
                    # shape => [C, t_size, C]
                    repeated_log_err   = w_log_err.unsqueeze(1).expand(C, t_size, C)
                    repeated_log_data  = log_w_dataset.unsqueeze(0).expand(C, t_size, C)
                    sum_ = repeated_log_err + repeated_log_data
                    # shape => [C, t_size]
                    logsumexp_c = torch.logsumexp(sum_, dim=2)

                    # Now for each t, we do sum_k [ p(t,k)* logsumexp_c[k,t] ] * s_{t,w}.
                    # shape => [t_size, C]
                    logsumexp_c_tk = logsumexp_c.transpose(0,1)  # [t_size, C]
                    weighted_by_p = logsumexp_c_tk * sub_predict
                    sum_over_k = weighted_by_p.sum(dim=1)  # shape [t_size]

                    # s_{t,w} factor:
                    s_factor = sub_conf[:, wb]  # shape [t_size]

                    # derivative wrt alpha is + s_{t,w} times that log(...) if exponent used. 
                    # so partial_grad[t] = s_{t,w} * sum_over_k[t] if that example actually uses w
                    # But there's no sign change for alpha, so we just add them.
                    partial_grad += torch.sum(s_factor * sum_over_k)

                    batch_grad[wb] += partial_grad

            # gradient update
            updated_alpha = alpha[start_w:end_w] + self.skill_lr * batch_grad
            updated_alpha = torch.clamp(updated_alpha, min=0.0)
            alpha[start_w:end_w] = updated_alpha

        self.worker_skill = alpha

    # -------------------------------------------------------------------------
    # Difficulty-step: update beta_t by gradient.
    #   exponent_{t,w} = (alpha_w - beta_t)* s_{t,w}.
    #   derivative wrt beta_t is - s_{t,w}.
    # -------------------------------------------------------------------------
    def _update_task_difficulty(self, predict_label, error_rates):
        device = self.device
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        alpha = self.worker_skill
        beta  = self.task_difficulty
        log_error_rates = torch.log(error_rates + 1e-10)

        grad_beta = torch.zeros(T, dtype=torch.float32, device=device)

        # We'll chunk over tasks to keep memory usage bounded
        for start_t in range(0, T, self.batch_size):
            end_t = min(start_t + self.batch_size, T)
            t_size = end_t - start_t

            # partial derivative for each t in this batch
            batch_grad = torch.zeros(t_size, device=device)

            sub_dataset = self.dataset_tensor[start_t:end_t]      # [t_size, W, C]
            sub_conf    = self.confidence_scores[start_t:end_t]   # [t_size, W]
            sub_predict = predict_label[start_t:end_t]            # [t_size, C]

            for w in range(W):
                aw = alpha[w]
                w_log_err = log_error_rates[w]

                # compute log( sum_c( dataset[t,w,c] * exp( w_log_err[k,c] ) ) ) for each t,k
                # shape => [t_size, C]
                batch_log_dataset = torch.log(sub_dataset[:, w, :] + 1e-10)

                # shape => [C, t_size, C]
                repeated_log_err = w_log_err.unsqueeze(1).expand(C, t_size, C)
                repeated_log_data = batch_log_dataset.unsqueeze(0).expand(C, t_size, C)
                sum_ = repeated_log_err + repeated_log_data
                # shape => [C, t_size]
                partial = torch.logsumexp(sum_, dim=2)
                # transpose => [t_size, C]
                partial_tk = partial.transpose(0,1)
                
                # sum over k with p(t,k)
                weighted_by_p = partial_tk * sub_predict
                # shape => [t_size]
                sum_over_k = weighted_by_p.sum(dim=1)

                # derivative wrt beta_t is - s_{t,w} times sum_over_k
                s_factor = sub_conf[:, w]
                # so the partial derivative for each t' is - s_{t',w} * sum_over_k[t']
                # we add that up across w
                batch_grad -= s_factor * sum_over_k

            grad_beta[start_t:end_t] += batch_grad

        # gradient step
        updated_beta = beta + self.skill_lr * grad_beta
        updated_beta = torch.clamp(updated_beta, min=0.0)
        self.task_difficulty = updated_beta

    # -------------------------------------------------------------------------
    # _get_likelihood: Weighted + skill- & difficulty-scaled log-likelihood
    # We do this for monitoring. 
    #   exponent_{t,w} = (alpha_w - beta_t)* s_{t,w}.
    # -------------------------------------------------------------------------
    def _get_likelihood(self, predict_label, error_rates):
        T = self.task_num
        W = self.worker_num
        C = self.class_num

        alpha = self.worker_skill
        beta  = self.task_difficulty
        log_error_rates = torch.log(error_rates + 1e-10)

        # prior
        marginal_probability = torch.sum(predict_label, dim=0)/T
        log_marginal_probability = torch.log(marginal_probability + 1e-10)

        log_L = 0.0
        batch_size = self.batch_size
        for start_t in range(0, T, batch_size):
            end_t = min(start_t + batch_size, T)
            t_size = end_t - start_t
            batch_dataset = self.dataset_tensor[start_t:end_t]  # [t_size, W, C]
            batch_conf = self.confidence_scores[start_t:end_t]  # [t_size, W]
            sub_predict = predict_label[start_t:end_t]          # [t_size, C]
            sub_beta = beta[start_t:end_t]                      # [t_size]

            # shape => [t_size, C]
            log_class_likelihood = torch.zeros((t_size, C), device=batch_dataset.device)
            
            for w in range(W):
                aw = alpha[w]
                # exponent_{t,w} = (aw - sub_beta[t']) * batch_conf[t', w]
                # shape => [t_size]
                exponent_vec = (aw - sub_beta) * batch_conf[:, w]

                w_log_err = log_error_rates[w]
                batch_log_dataset = torch.log(batch_dataset[:, w, :] + 1e-10)  # [t_size, C]

                repeated_log_err = w_log_err.unsqueeze(1).expand(C, t_size, C)
                repeated_log_data = batch_log_dataset.unsqueeze(0).expand(C, t_size, C)
                sum_ = repeated_log_err + repeated_log_data
                partial = torch.logsumexp(sum_, dim=2)  # [C, t_size]
                partial_tk = partial.transpose(0,1)     # [t_size, C]

                # multiply by exponent_vec[t']
                weighted = partial_tk * exponent_vec.unsqueeze(1)

                log_class_likelihood += weighted

            # Add prior
            log_posterior = log_marginal_probability.unsqueeze(0) + log_class_likelihood
            log_L += torch.sum(torch.logsumexp(log_posterior, dim=1))

        return log_L.item()

