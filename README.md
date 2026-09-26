# Machine Learning Algorithms from Scratch

Educational implementations of classical machine-learning algorithms using
NumPy and pandas for the model logic. Each algorithm is developed and explained
in a Jupyter notebook, then evaluated on a small public or synthetic dataset.

The repository focuses on understanding the mechanics of optimization,
distance metrics, regularization, and prediction rather than replacing
production libraries such as scikit-learn.

## Implemented algorithms

| Algorithm | Notebook | Highlights |
| --- | --- | --- |
| Linear regression | `Linear_models/Linear_regression.ipynb` | Batch or stochastic gradient descent, configurable learning rate, L1/L2/Elastic Net regularization, and MAE/MSE/RMSE/MAPE/R2 metrics. |
| k-NN classification | `Metric_algorithms/KNN_classification.ipynb` | Binary classification, probability estimates, four distance metrics, and uniform/rank/distance weighting. |
| k-NN regression | `Metric_algorithms/KNN_regression.ipynb` | Regression with uniform, rank-based, or inverse-distance weighting. |

Supported k-NN distances:

- Euclidean;
- Manhattan;
- Chebyshev;
- cosine distance.

## Datasets

- linear regression: a synthetic regression dataset generated with
  `sklearn.datasets.make_regression`;
- k-NN classification: the UCI Banknote Authentication dataset included as
  `data_banknote_authentication.txt`;
- k-NN regression: the scikit-learn diabetes dataset.

scikit-learn is used only to load/split demonstration datasets; the model
implementations themselves use NumPy and pandas.

## Quick start

Create an environment and install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas scikit-learn
jupyter lab
```

Open any notebook and run its cells from top to bottom. The notebooks contain
the implementation, explanation, and a compact experiment in one place.

## Repository structure

```text
.
├── Linear_models/
│   └── Linear_regression.ipynb
├── Metric_algorithms/
│   ├── KNN_classification.ipynb
│   └── KNN_regression.ipynb
└── data_banknote_authentication.txt
```

## Scope and limitations

This is a learning repository, not a drop-in estimator package. The notebooks
prioritize readable implementations and expose intermediate calculations. They
do not yet provide a shared package API, automated tests, or performance
optimizations for large datasets.

## Roadmap

Natural extensions include decision trees, ensemble methods, clustering, and
dimensionality reduction. These are roadmap items and are not claimed as
implemented in the current repository.
