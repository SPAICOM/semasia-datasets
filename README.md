# REPO TITLE


<h5 align="center">
    
[![arXiv](https://img.shields.io/badge/Arxiv-2605.09485-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2605.09485)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/semasia-datasets/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> Latent representations learned by neural networks often exhibit semantic structure, where concept similarity is reflected by geometric proximity in embedding space. However, comparing such spaces across models remains difficult: changes in architecture, pretraining data, objective, or random seed can yield embeddings with similar content but incompatible geometry. This latent space alignment problem is central to interpretability, transfer and multimodal learning, federated systems, and semantic communication; however, progress remains limited by the lack of large-scale, model-diverse, and metadata-rich benchmarks. To address this gap, we introduce SEMASIA, a large-scale collection of latent representations extracted from approximately 1,700 pretrained vision models across eight standard image-classification benchmarks. SEMASIA pairs embeddings with structured metadata describing architectures, training regimes, pretraining sources, and model scale. We demonstrate three applications of the resource. First, we analyze the conceptual organization of individual latent spaces, showing consistent prototype-like clustering and hierarchical semantic neighborhoods across models and datasets. Second, we benchmark supervised alignment mappings between latent spaces using reconstruction error and downstream task performance. Third, we perform a large-scale regression analysis of how pretraining-data complexity, specialization, transfer learning, augmentation, and model scale relate to geometric and probing properties of embeddings. By coupling representational scale with standardized metadata, SEMASIA provides a reproducible foundation for studying latent geometry, evaluating alignment methods, and developing next-generation heterogeneous and interoperable AI systems. 

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management and [`just`](https://github.com/casey/just) as the task runner.

### Install prerequisites

Install the required tools:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [`just`](https://github.com/casey/just)

Follow the installation instructions from their official documentation.

### Setup the development environment

From the project root, run:

```bash
just setup
```

The `setup` recipe will:

- Create the `.venv` virtual environment (if it does not exist)
- Install all project dependencies using `uv`

After the command completes, the development environment will be ready to use. 🚀

## Analysis

All analyses are run via `just`. Configurations are managed with [Hydra](https://hydra.cc) — any default can be overridden by appending `key=value` pairs to the command (e.g. `just tda-compare cifar10 comparison.distance=wasserstein`).

### Encoding

Before running any analysis on a new dataset, encode it with all available timm models:

```bash
just timm-encode DATASET   # encode locally without pushing to HuggingFace
just push DATASET          # encode + push to HuggingFace + update readme
```

### Statistical Metrics

Compute geometric and probing properties of latent spaces, then run regression analysis over model metadata:

```bash
just compute-metrics DATASET        # compute stat metrics for a dataset
just compute-metrics-tda DATASET    # include TDA metrics (slower)
just compute-metrics-test DATASET   # quick test run limited to 5 models

# Two-phase workflow for large runs
just compute-download               # Phase 1: download and cache all latents
just compute-compute                # Phase 2: compute metrics from cached latents

just stat                           # run regression analysis on precomputed metrics
just stat-plot                      # plot regression results
```

### Graph Signatures

Extract KNN-graph structural metrics (cycles, Wiener index, eigengap, etc.) from latent spaces:

```bash
just tsp-extraction
```

### Prototype Analysis

Compare prototype structures across pairs of models:

```bash
just proto-compare                             # default model pair from config
just proto-compare-dataset DATASET            # override dataset
just proto-compare-models MODEL_A MODEL_B     # override model pair
```

### Latent Space Alignment

Evaluate alignment methods (proto, CCA, linear) that transmit embeddings from model A into model B's space, and visualize the results:

```bash
just alignment                                 # run evaluation with config defaults
just alignment-plot                            # plot accuracy/MSE vs compression ratio (k)

# PC correlation heatmap between two models on a dataset
just plot-heatmap MODEL_A MODEL_B DATASET
```

Prototype-based alignment using lstsq probing on the transmitted space:

```bash
just proto-alignment                           # default config
just proto-alignment-dataset DATASET          # override dataset
just proto-alignment-models MODEL_A MODEL_B   # override model pair
```

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@misc{pandolfo2026semasialargescaledatasetsemantically,
      title={SEMASIA: A Large-Scale Dataset of Semantically Structured Latent Representations}, 
      author={Mario Edoardo Pandolfo and Enrico Grimaldi and Lorenzo Marinucci and Leonardo Di Nino and Simone Fiorellino and Sergio Barbarossa and Paolo Di Lorenzo},
      year={2026},
      eprint={2605.09485},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.09485}, 
}
```

## Authors

- [Mario Edoardo Pandolfo](https://scholar.google.com/citations?user=wAeScL8AAAAJ)
- [Enrico Grimaldi](https://scholar.google.com/citations?user=Y-31eCwAAAAJ)
- [Lorenzo Marinucci](https://scholar.google.com/citations?user=_VdGlLoAAAAJ)
- [Leonardo Di Nino](https://scholar.google.com/citations?user=4UdFEvAAAAAJ)
- [Simone Fiorellino](https://scholar.google.com/citations?user=nKMc4GQAAAAJ)
- [Sergio Barbarossa](https://scholar.google.com/citations?user=2woHFu8AAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?user=VZYvspQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-89CFF0?style=for-the-badge&logo=hyperland&logoColor=white)
![w&b](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
