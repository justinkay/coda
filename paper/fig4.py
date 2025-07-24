from coda.datasets import Dataset
from coda.options import LOSS_FNS
from coda.oracle import Oracle
import os
from coda.coda import CODA
import torch
import sklearn

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.serif"] = ["Nimbus Roman"]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(10,2))

# civilcomments
dataset = Dataset(os.path.join('../data', "civilcomments.pt"), device='cuda')
loss_fn = LOSS_FNS['acc']
oracle = Oracle(dataset, loss_fn=loss_fn)
true_losses = oracle.true_losses(dataset.preds)
true_losses.shape
selector = CODA(dataset)
prob_best = selector.get_pbest().squeeze()
pred_best = torch.zeros_like(prob_best)
pred_best[torch.argmax(prob_best)] = prob_best[torch.argmax(prob_best)]
true_best = torch.zeros_like(prob_best)
true_best[torch.argmin(torch.tensor(true_losses))] = prob_best[torch.argmin(torch.tensor(true_losses))]
true_accs = 1 - true_losses

# CONFUSION MATRIX 1 - Predicted best
# pbest_idx = torch.argmax(prob_best)
# pworst_idx = torch.argmin(prob_best)
best_idx = torch.argmax(true_accs)
# worst_idx = torch.argmin(true_accs)
# pbest_preds = torch.argmax(dataset.preds[pbest_idx], dim=-1)
# cm = sklearn.metrics.confusion_matrix(oracle.labels.cpu(), pbest_preds.cpu().numpy(), normalize='true')
# disp = sklearn.metrics.ConfusionMatrixDisplay(cm)
# disp.plot(ax=ax1, colorbar=False)
# ax1.set_title("Selected model, time 0",fontsize=16)

# CONFUSION MATRIX 2 - True best
best_preds = torch.argmax(dataset.preds[best_idx], dim=-1)
cm = sklearn.metrics.confusion_matrix(oracle.labels.cpu(), best_preds.cpu().numpy(), normalize='true')
disp = sklearn.metrics.ConfusionMatrixDisplay(cm)
disp.plot(ax=ax1, colorbar=False)
ax1.set_title("True best model",fontsize=16)
ax1.set_xlabel("Predicted label",fontsize=14)
ax1.set_ylabel("True label",fontsize=14)

# CLASS MARGINAL
true_marginal = torch.bincount(oracle.labels,minlength=len(selector.pi_hat)).cpu().float()
true_marginal /= true_marginal.sum()
ax2.bar([ l - 0.2 for l in list(range(len(selector.pi_hat)))  ], true_marginal.cpu(), width=0.4, label="True")
ax2.bar( [ l + 0.2 for l in list(range(len(selector.pi_hat)))  ], selector.pi_hat.cpu(), width=0.4, label="Est.")
ax2.set_xticks([0,1])
ax2.set_xlabel("Class idx",fontsize=14)
ax2.legend(fontsize=14, bbox_to_anchor=(0.17,0.53), loc='lower left')
ax2.set_title("Class dist.",fontsize=16)
ax2.set_ylabel("Class proportion",fontsize=14)

# -----------------------------

# GLUE COLA
dataset = Dataset(os.path.join('../data', "glue_cola.pt"),  device='cuda' )
oracle = Oracle(dataset, loss_fn=loss_fn)
true_losses = oracle.true_losses(dataset.preds)
true_losses.shape
selector = CODA(dataset)
true_accs = 1 - true_losses


# CONFUSION MATRIX 1 - Predicted best
pbest_idx = 31 # we know this is the problematic one from testing
best_idx = torch.argmax(true_accs)
pbest_preds = torch.argmax(dataset.preds[pbest_idx], dim=-1)
cm = sklearn.metrics.confusion_matrix(oracle.labels.cpu(), pbest_preds.cpu().numpy(), normalize='true')
disp = sklearn.metrics.ConfusionMatrixDisplay(cm)
disp.plot(ax=ax3, colorbar=False)
ax3.set_title("Selected model, time 70",fontsize=16)
ax3.set_xlabel("Predicted label",fontsize=14)
ax3.set_ylabel("True label",fontsize=14)

# CONFUSION MATRIX 2 - True best
# best_preds = torch.argmax(dataset.preds[best_idx], dim=-1)
# cm = sklearn.metrics.confusion_matrix(oracle.labels.cpu(), best_preds.cpu().numpy(), normalize='true')
# disp = sklearn.metrics.ConfusionMatrixDisplay(cm)
# disp.plot(ax=ax2, colorbar=False)
# ax2.set_title("True best model",fontsize=16)

# CLASS MARGINAL
true_marginal = torch.bincount(oracle.labels,minlength=len(selector.pi_hat)).cpu().float()
true_marginal /= true_marginal.sum()
ax4.bar([ l - 0.2 for l in list(range(len(selector.pi_hat)))  ], true_marginal.cpu(), width=0.4, label="True")
ax4.bar( [ l + 0.2 for l in list(range(len(selector.pi_hat)))  ], selector.pi_hat.cpu(), width=0.4, label="Est.")
ax4.set_xticks([0,1])
ax4.set_xlabel("Class idx",fontsize=14)
ax4.legend(fontsize=14,loc='lower right', bbox_to_anchor=(1.05,-0.05))
ax4.set_title("Class dist.",fontsize=16)
ax4.set_ylabel("Class proportion",fontsize=14)

plt.suptitle("CivilComments: Failure Case", fontsize=17, y=1.13, x=0.3)
plt.text(.7, 1.03, 'CoLA: Failure Case', transform=fig.transFigure, horizontalalignment='center', fontsize=17)
plt.subplots_adjust(wspace=0.45, hspace=0.5)

plt.savefig('fig4_cameraready.pdf', dpi=300, bbox_inches='tight')