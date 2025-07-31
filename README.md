# Consensus-Driven Active Model Selection 

Documentation coming soon!

## Benchmark dataset

[Data download (3.25GB)](https://drive.google.com/file/d/1H8zXwAGkkAQP5L1gofpeF69jeIZqzPaW/view?usp=sharing)

## Reference

### Consensus-Driven Active Model Selection

[Justin Kay](https://justinkay.github.io), [Grant Van Horn](https://gvanhorn38.github.io/), [Subhransu Maji](https://people.cs.umass.edu/~smaji/), [Daniel Sheldon](https://people.cs.umass.edu/~sheldon/) and [Sara Beery](https://beerys.github.io/).

The widespread availability of off-the-shelf machine learning models poses a challenge: which model, of the many available candidates, should be chosen for a given data analysis task? This question of model selection is traditionally answered by collecting and annotating a validation dataset---a costly and time-intensive process. We propose a method for active model selection, using predictions from candidate models to prioritize the labeling of test data points that efficiently differentiate the best candidate. Our method, CODA, performs consensus-driven active model selection by modeling relationships between classifiers, categories, and data points within a probabilistic framework. The framework uses the consensus and disagreement between models in the candidate pool to guide the label acquisition process, and Bayesian inference to update beliefs about which model is best as more information is collected. We validate our approach by curating a collection of 26 benchmark tasks capturing a range of model selection scenarios. CODA outperforms existing methods for active model selection significantly, reducing the annotation effort required to discover the best model by upwards of 70% compared to the previous state-of-the-art.

ICCV 2025 Highlight.
