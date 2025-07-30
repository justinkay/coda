import torch
from torch.distributions import Beta
import torch.nn.functional as F
from tqdm import tqdm
import random
import mlflow

from coda.base import ModelSelector
from coda.util import Ensemble, _check, _check_prob, plot_bar

_DEBUG = True       # check for valid PDFs, infs, NaNs
_DEBUG_VIZ = False  # plot PBest and EIG every iter


def dirichlet_to_beta(alpha_dirichlet: torch.Tensor):
    """
    Get parameters for beta distributions representing the diagonal.
    Args:
        alpha_dirichlet: shape (..., H, C, C)
    Returns:
        alpha_cc, beta_cc: shape (..., H, C)
    """
    C = alpha_dirichlet.shape[-1]
    alpha_cc = alpha_dirichlet[..., torch.arange(C), torch.arange(C)]
    beta_cc  = alpha_dirichlet.sum(dim=-1) - alpha_cc
    return alpha_cc, beta_cc


def create_confusion_matrices(true_labels: torch.Tensor,
                              model_predictions: torch.Tensor,
                              mode='hard') -> torch.Tensor:
    H, N, C = model_predictions.shape
    dev = model_predictions.device
    true_one_hot = F.one_hot(true_labels, C).float().to(dev)

    if mode == 'hard':
        preds = F.one_hot(model_predictions.argmax(-1), C).float()
    elif mode == 'soft':
        preds = model_predictions
    else:
        raise ValueError(mode)

    conf = torch.einsum('nc, hnj -> hcj', true_one_hot, preds)
    return conf / conf.sum(-1, keepdim=True).clamp_min(1e-6)


def initialize_dirichlets(soft_confusion: torch.Tensor,
                          prior_strength: float,
                          disable_diag_prior=False) -> torch.Tensor:
    H, C, _ = soft_confusion.shape

    if disable_diag_prior:
        # uniform - 2 pseudo counts per row to match diag method
        base = torch.full((C, C), 2 / C,
                          dtype=soft_confusion.dtype,
                          device=soft_confusion.device)
    else:
        base = torch.full((C, C), 1.0 / (C - 1),
                            dtype=soft_confusion.dtype,
                            device=soft_confusion.device)
        base.fill_diagonal_(1.0)

    base = base.unsqueeze(0).expand(H, C, C)
    return base + prior_strength * soft_confusion


def batch_update_dirichlet_for_item(dirichlet_alphas: torch.Tensor,
                                    classifier_preds: torch.Tensor,
                                    update_weight: float = 1.0) -> torch.Tensor:
    N, H, C = classifier_preds.shape
    updated = dirichlet_alphas[None, None].expand(N, C, H, C, C).clone()
    updates = classifier_preds[:, None].expand(-1, C, -1, -1) * update_weight
    for c in range(C):
        updated[:, c, :, c, :] += updates[:, c, :, :]
    return updated


def compute_pbest_beta_batched(alpha_batch: torch.Tensor,  # (B_, C_, C, H)
                                beta_batch:  torch.Tensor, # (B_, C_, C, H)
                                num_points: int = 256,
                                eps: float = 1e-30,
                                chunk_size: int = None) -> torch.Tensor:
    device = alpha_batch.device
    N = alpha_batch.shape[0]
    C, H = alpha_batch.shape[-2:]
    chunk_size = chunk_size or N
    x = torch.linspace(1e-6, 1 - 1e-6, num_points, device=device).unsqueeze(-1) # P×1

    prob_out = torch.zeros_like(alpha_batch)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        a_flat = alpha_batch[start:end].reshape(-1, H)          # B_*C_*C × H
        b_flat = beta_batch[start:end].reshape(-1, H)
        
        logpdf = Beta(a_flat.reshape(-1), b_flat.reshape(-1)).log_prob(x)
        pdf = logpdf.exp().T.reshape(-1, H, num_points)         # B_*C_*C × H × P
        if _DEBUG: _check(pdf, "pdf")

        cdf = torch.zeros_like(pdf)
        for j in range(1, num_points):
            dx = x[j] - x[j-1]
            cdf[:, :, j] = cdf[:, :, j-1] + 0.5*(pdf[:, :, j] + pdf[:, :, j-1])*dx
        if _DEBUG: _check(cdf, "cdf")

        log_cdf = torch.log(cdf.clamp_min(eps))
        # clamp to min/max float32 +-(log(3.4 * 1e38) = ~88) to avoid inf; 
        # rare that this happens (only observed with uniform prior)
        prod_excl = torch.exp( (log_cdf.sum(1, keepdim=True) - log_cdf).clamp(-80,80) ) 
        integrand = pdf * prod_excl
        if _DEBUG: _check(integrand, "integrand") # probably prod_excl's fault if bad

        prob = torch.trapz(integrand, x.squeeze(), dim=2)
        if _DEBUG: _check(prob, "Pbest(beta)") 

        prob = prob / prob.sum(-1, keepdim=True).clamp_min(eps)
        if _DEBUG: _check_prob(prob, "Pbest(beta) normalized")

        prob_out[start:end] = prob.reshape(alpha_batch[start:end].shape)

    return prob_out # (B_, C_, C, H)


def pbest_row_mixture_batched(updated_dirichlet: torch.Tensor,
                                pi_hat: torch.Tensor,
                                num_points: int = 256) -> torch.Tensor:
    """
    Args:
        updated_dirichlet: (B_, C_, H, C, C)
        pi: (C,)
    where:
        B_ and C_ are additional dimensions that can be used for hypothetical item and hypothetical class updates, respectively
            (i.e. all operations are broadcast over B_ and C_)
        pi is the marginal class distribution P(class=C) over the entire dataset
    
    Returns:
        prob_best: (B_, C_, H),  P(h is best | C_, B_)
    """
    C = updated_dirichlet.shape[-1]

    # P(h is best | row c)
    alpha_cc, beta_cc = dirichlet_to_beta(updated_dirichlet)
    prob_best_b_c_ch = compute_pbest_beta_batched(alpha_cc.transpose(-1, -2), beta_cc.transpose(-1, -2), num_points=num_points) # (B_,C_,C,H)

    # convert conditional to marginal probabilities using pi_hat
    # expected P(best | item b) = Σ_c expected P(best | item b, class=c) * P(class=c)
    marginal_probs = (prob_best_b_c_ch * pi_hat.view(1, C, 1)).sum(-2)  # (B_, C_, H)

    return marginal_probs


def batch_update_beta(selector, # selector.dirichlets: (H,C,C)
                      preds,    # (B, H)
                      update_weight=1.0
                      ): 
    B, H = preds.shape
    C = selector.dirichlets.shape[-1]
    alpha_cc_before, beta_cc_before = dirichlet_to_beta(selector.dirichlets) # (H, C)

    pred_classes = preds.unsqueeze(1).expand(B,C,H)
    class_range  = torch.arange(C, device=alpha_cc_before.device).unsqueeze(1).expand(B,C,H)
    eq_mask = (pred_classes == class_range) # B,C,H
    eq_mask = eq_mask.permute(0,2,1) # B,H,C

    alpha_batch = alpha_cc_before.expand(B, H, C).clone()
    beta_batch = beta_cc_before.expand(B, H, C).clone()
    alpha_batch[eq_mask] += 1.0 * update_weight
    beta_batch[~eq_mask]  += 1.0 * update_weight

    return alpha_batch, beta_batch # (B, H, C), (B, H, C)


class CODA(ModelSelector):
    def __init__(self, 
                 dataset,
                 prefilter_n=0,
                 alpha=0.9,
                 learning_rate=0.01,
                 multiplier=2.0,
                 disable_diag_prior=False,  # for ablation 1
                 q='eig',                   # for ablation 2
                 ):
        self.dataset = dataset
        self.device = dataset.preds.device
        self.H, self.N, self.C = dataset.preds.shape
        self.prefilter_n = prefilter_n
        self.disable_diag_prior = disable_diag_prior
        self.q = q

        # hyperparams
        self.prior_strength = (1 - alpha)
        self.update_strength = learning_rate

        # initialize dirichlets
        ens_pred = Ensemble(dataset.preds).get_preds()
        ens_pred_hard = ens_pred.argmax(-1)  # pseudo labels
        soft_conf = create_confusion_matrices(ens_pred_hard, dataset.preds, mode='soft')
        self.dirichlets = multiplier * initialize_dirichlets(soft_conf, self.prior_strength, self.disable_diag_prior)
        self.update_pi_hat()

        self.labeled_idxs, self.labels = [], []
        self.unlabeled_idxs = list(range(self.N))
        self.q_vals = []
        self.stochastic = False
        self.step = 0

    @classmethod
    def from_args(cls, dataset, args):
        return cls(dataset,
                   prefilter_n=args.prefilter_n,
                   alpha=args.alpha,
                   learning_rate=args.learning_rate,
                   multiplier=args.multiplier,
                   disable_diag_prior=args.no_diag_prior,
                   q=args.q)

    def _prefilter(self, idxs):
        # filter any data points where every model disagrees - waste of compute
        maj, _ = torch.mode(self.dataset.preds.argmax(-1), dim=0)
        mask = (self.dataset.preds.argmax(-1) != maj).sum(0) > 0
        idxs = [i for i in idxs if mask[i]]
        # can also randomly subsample (disabled by default)
        if self.prefilter_n and len(idxs) > self.prefilter_n:
            idxs = random.sample(idxs, self.prefilter_n)
            self.stochastic = True
        return idxs

    def update_pi_hat(self):
        adjusted = torch.einsum('hcs, hns -> hnc', self.dirichlets, self.dataset.preds)
        # per item
        self.pi_hat_xi = adjusted.sum(0)
        self.pi_hat_xi = self.pi_hat_xi / self.pi_hat_xi.sum(dim=-1, keepdim=True).clamp_(min=1e-12)
        # marginal (entire dataset)
        self.pi_hat = self.pi_hat_xi.sum(0)
        self.pi_hat = self.pi_hat / self.pi_hat.sum()

    def eig_batched(self, chunk_size: int = 100, update_weight: float = 1.0, num_points: int = 256):
        """
            TODO: Document shapes etc.
        """
        candidate_ids = self._prefilter(self.unlabeled_idxs) or self.unlabeled_idxs
        classifier_preds = self.dataset.preds.permute(1, 0, 2)
        candidates = torch.tensor(candidate_ids, device=classifier_preds.device)
        N, H, C = classifier_preds.shape

        # compute current pbest per row
        dirichlets_before = self.dirichlets.unsqueeze(0).unsqueeze(0).expand(1, 1, H, C, C)
        
        # get diagonal betas
        alpha_cc_before, beta_cc_before = dirichlet_to_beta(dirichlets_before) # (1, 1, H, C)
        alpha_cc_before = alpha_cc_before.permute(0,3,1,2)  # (1, C, 1, H)
        beta_cc_before  = beta_cc_before.permute(0,3,1,2)   # (1, C, 1, H)
        pbest_rows_before = compute_pbest_beta_batched(alpha_cc_before, beta_cc_before).squeeze(-2) # (1, C, H)

        mixture0 = (self.pi_hat[:, None] * pbest_rows_before).sum(1)   # (1,H)
        H_before = -(mixture0.clamp_min(1e-12).mul(mixture0.clamp_min(1e-12).log2())).sum(-1)

        # broadcast helpers
        mixture0_bc = mixture0.view(1, 1, H)      # (1,1,H)
        pi_hat_row  = self.pi_hat.view(1, C, 1)   # (1,C,1)

        eig_chunks = []
        for s in tqdm(range(0, len(candidates), chunk_size)):
            ids   = candidates[s:s + chunk_size] # (B,)
            preds = classifier_preds[ids].argmax(-1) # (B, H)
            pi_hat_xi = self.pi_hat_xi[ids]

            # do all hypothetical updates at once
            alpha_hypothetical, beta_hypothetical = batch_update_beta(self, preds, update_weight) # (B,H,C_)
            alpha_hypothetical = alpha_hypothetical.permute(0,2,1).unsqueeze(-2)  # (B, C_, 1, H)
            beta_hypothetical = beta_hypothetical.permute(0,2,1).unsqueeze(-2)    # (B, C_, 1, H)

            pbest_hypothetical_rows = compute_pbest_beta_batched(alpha_hypothetical, 
                                                                    beta_hypothetical, 
                                                                    num_points=num_points).squeeze(-2) # (B, C_, H)
            deltas = pi_hat_row * (pbest_hypothetical_rows - pbest_rows_before) # (B,C,H)
            mix_new = mixture0_bc + deltas # (B,C,H)
            H_after = -(mix_new.clamp_min(1e-12).mul(mix_new.clamp_min(1e-12).log2())).sum(-1) # (B,C)
            
            eig = H_before - (pi_hat_xi * H_after).sum(-1) # (B,)
            eig_chunks.append(eig)

        return torch.cat(eig_chunks), candidate_ids

    def get_next_item_to_label(self):
        if self.q == 'eig':
            # default; expected information gain
            q_vals, cand = self.eig_batched()
        elif self.q == 'iid':
            # random sampling (used for ablation 2)
            cand = self._prefilter(self.unlabeled_idxs) or self.unlabeled_idxs
            q_vals = 1/len(cand) * torch.ones(len(cand), device=self.device)
        elif self.q == 'uncertainty':
            # uncertainty-based sampling (used for ablation 2)
            from coda.baselines.uncertainty import uncertainty
            cand = self._prefilter(self.unlabeled_idxs) or self.unlabeled_idxs
            q_vals = uncertainty(self.dataset.preds, cand)
        else:
            raise NotImplementedError(self.q)

        if _DEBUG_VIZ:
            img = plot_bar(q_vals)
            mlflow.log_image(img, key="EIG", step=self.step)
            img.save('eig-'+str(self.step)+'.png')

        # greedy sampling with random selection between ties
        best = q_vals.max()
        ties = torch.isclose(q_vals, best, rtol=1e-8)
        idx_local = random.choice(torch.nonzero(ties, as_tuple=True)[0].tolist()) \
                    if ties.sum() > 1 else torch.argmax(q_vals).item()
        if ties.sum() > 1:
            self.stochastic = True

        return cand[idx_local], q_vals[idx_local].item()

    def add_label(self, idx, true_class, selection_prob):
        preds = F.one_hot(self.dataset.preds[:, idx].argmax(-1), self.C).float()
        self.dirichlets[:, true_class] += self.update_strength * preds

        self.update_pi_hat()
        self.labeled_idxs.append(idx)
        self.labels.append(int(true_class))
        self.q_vals.append(selection_prob)
        self.unlabeled_idxs.remove(idx)

    def get_pbest(self):
        H, C, _ = self.dirichlets.shape
        expanded = self.dirichlets.unsqueeze(0).unsqueeze(0).expand(1, 1, H, C, C)

        pbest = pbest_row_mixture_batched(expanded, self.pi_hat).squeeze(0) # (H,)
        if _DEBUG: _check(pbest, "Pbest") 

        return pbest

    def get_best_model_prediction(self):
        pbest = self.get_pbest()
        
        if _DEBUG_VIZ:
            img = plot_bar(pbest)
            mlflow.log_image(img, key="PBest", step=self.step)
            img.save('pbest-'+str(self.step)+'.png')

        # track how many times we've done this
        self.step += 1 

        return torch.argmax(pbest)
