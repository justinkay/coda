# Consensus-Driven Active Model Selection 

[ [arXiv](https://www.arxiv.org/abs/2507.23771) ] [ [Demo](https://huggingface.co/spaces/justinkay/coda) ]

The widespread availability of off-the-shelf machine learning models -- for instance, the more than 2M currently available on [HuggingFace Models](https://huggingface.co/models?sort=trending) -- poses a challenge: which model, of the many available candidates, should be chosen for a given data analysis task?

We introduce <b>CODA</b>, a <b>co</b>nsensus-<b>d</b>riven method for <b>a</b>ctive model selection, to answer this question as efficiently as possible:

<br>
<img width="14575" height="4505" alt="coda-cameraready" src="https://github.com/user-attachments/assets/77a4c0fc-9ca4-4c2c-a6c3-f110b774ea7f" />
<br>
<br>
 
<b>CODA</b> uses the consensus and disagreement between models in the candidate pool to guide
the label acquisition process, and Bayesian inference to update beliefs about which model is best as more information is collected. <b>CODA</b> outperforms existing methods for active model selection significantly, 
reducing the annotation effort required to discover the best model by upwards of 70% compared to the previous state-of-the-art.


## Install

**1. Install PyTorch and torchvision.** Follow the [official install guide](https://pytorch.org/get-started/locally/) to install the correct versions for your CUDA version or CPU.

**2. Install CODA.** Clone this repository and run:

```bash
pip install -e .
```

## Dataset download

[Data download (3.25GB)](https://drive.google.com/file/d/1H8zXwAGkkAQP5L1gofpeF69jeIZqzPaW/view?usp=sharing)

## Run an active model selection experiment

To run 5 random seeds of <b>CODA</b> with default hyperparameters on CIFAR10-high:

```python main.py --task cifar10_5592 --method coda```

See [main.py](main.py) for the full list of command line options.

## View results

Results are saved to a SQLite database managed by MLFlow. See scripts in `paper/` for how to query the database to summarize results. 

You can also visualize results through the MLFlow UI using `mlflow ui --backend-store-uri sqlite:///coda.sqlite`, however you will need to first aggregate results from different seeds by running `python scripts/aggregate_results.py`.

## Reference

### Consensus-Driven Active Model Selection

[Justin Kay](https://justinkay.github.io), [Grant Van Horn](https://gvanhorn38.github.io/), [Subhransu Maji](https://people.cs.umass.edu/~smaji/), [Daniel Sheldon](https://people.cs.umass.edu/~sheldon/) and [Sara Beery](https://beerys.github.io/).

The widespread availability of off-the-shelf machine learning models poses a challenge: which model, of the many available candidates, should be chosen for a given data analysis task? This question of model selection is traditionally answered by collecting and annotating a validation dataset---a costly and time-intensive process. We propose a method for active model selection, using predictions from candidate models to prioritize the labeling of test data points that efficiently differentiate the best candidate. Our method, <b>CODA</b>, performs <b>co</b>nsensus-<b>d</b>riven <b>a</b>ctive model selection by modeling relationships between classifiers, categories, and data points within a probabilistic framework. The framework uses the consensus and disagreement between models in the candidate pool to guide the label acquisition process, and Bayesian inference to update beliefs about which model is best as more information is collected. We validate our approach by curating a collection of 26 benchmark tasks capturing a range of model selection scenarios. CODA outperforms existing methods for active model selection significantly, reducing the annotation effort required to discover the best model by upwards of 70% compared to the previous state-of-the-art.

ICCV 2025 Highlight.
