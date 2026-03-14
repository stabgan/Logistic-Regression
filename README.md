# 📊 Logistic Regression — Social Network Ads Classifier

A binary classification project using **Logistic Regression** to predict whether a user will purchase a product based on their age and estimated salary. Implemented in both Python (scikit-learn) and R.

## 🧠 Methodology

1. **Data Loading** — Read the `Social_Network_Ads.csv` dataset containing user demographics and purchase history.
2. **Feature Selection** — Use `Age` and `EstimatedSalary` as predictor variables; `Purchased` (0/1) as the target.
3. **Train/Test Split** — 75% training, 25% testing (`random_state=0` for reproducibility).
4. **Feature Scaling** — Standardize features using `StandardScaler` so both dimensions contribute equally.
5. **Model Training** — Fit a Logistic Regression classifier on the scaled training data.
6. **Evaluation** — Generate a confusion matrix, accuracy score, and full classification report.
7. **Visualization** — Plot decision boundaries for both training and test sets.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Primary implementation |
| 📊 R | Alternative implementation |
| 🤖 scikit-learn | ML model, preprocessing, evaluation |
| 🔢 NumPy | Numerical operations |
| 🐼 pandas | Data loading and manipulation |
| 📈 matplotlib | Decision boundary visualization |

## 📦 Dependencies

### Python

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### R

```
caTools
ElemStatLearn
```

## 🚀 How to Run

### Python

```bash
cd Logistic-Regression
python logistic_regression.py
```

The script will print the confusion matrix, accuracy, and classification report to the console, then display the training and test set decision boundary plots.

### R

```bash
Rscript logistic_regression.R
```

## 📁 Project Structure

```
.
├── logistic_regression.py    # Python implementation
├── logistic_regression.R     # R implementation
├── Social_Network_Ads.csv    # Dataset
├── LICENSE                   # MIT License
└── README.md
```

## ⚠️ Known Issues

- The R script depends on `ElemStatLearn`, which has been archived from CRAN. You may need to install it from an archive mirror or use an alternative visualization approach.
- The visualization step generates a dense meshgrid and can be slow on large datasets.
- The dataset path is relative — run the script from the project root directory.

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
