# Logistic Regression — Social Network Ads

Binary classification using Logistic Regression to predict whether a user purchases a product based on age and estimated salary.

## What It Does

Trains a Logistic Regression model on the **Social Network Ads** dataset and visualises the decision boundary for both training and test sets. The workflow:

1. Loads the dataset (400 users with age, salary, and purchase label)
2. Splits into 75 / 25 train-test
3. Applies standard scaling
4. Fits a Logistic Regression classifier
5. Outputs a confusion matrix and two decision-boundary plots

Implementations are provided in both **Python** and **R**.

## Dataset

| Column           | Description                        |
|------------------|------------------------------------|
| User ID          | Unique identifier (not used)       |
| Gender           | Male / Female (not used)           |
| Age              | User age                           |
| EstimatedSalary  | Annual estimated salary            |
| Purchased        | 0 = No, 1 = Yes (target)          |

Source: `Social_Network_Ads.csv` (included in repo).

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.10+ | Primary language |
| 📊 scikit-learn | Logistic Regression, train/test split, metrics |
| 🔢 NumPy | Numerical operations |
| 🐼 pandas | Data loading |
| 📈 matplotlib | Decision-boundary visualisation |
| 📊 R | Alternative implementation |

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python logistic_regression.py
```

For the R version:

```r
source("logistic_regression.R")
```

> The R script requires the `caTools` and `ElemStatLearn` packages.

## Fixes Applied

The original code was written for scikit-learn < 0.18. The following modernisations were made:

- `sklearn.cross_validation` → `sklearn.model_selection` (removed in sklearn 0.20)
- All imports moved to top of file (PEP 8)
- Removed duplicate `ListedColormap` import
- Fixed `ListedColormap()(i)` scatter color — now uses plain color strings
- Added explicit `solver='lbfgs'` and `max_iter=200` to `LogisticRegression`
- Added `if __name__ == "__main__"` guard
- CSV path is now relative to script location via `os.path`
- Extracted plot logic into a reusable `_plot_decision_boundary()` function
- Added `requirements.txt`

## ⚠️ Known Issues

- The R script depends on `ElemStatLearn`, which has been archived on CRAN. Install from archive or use an alternative visualisation approach.
- Decision-boundary plots use a fine mesh (`step=0.01`) which can be slow on large feature ranges.

## License

MIT
