# Adult Income Prediction using Machine Learning

A supervised machine learning project that predicts whether a person's annual income is **above $50K** or **at or below $50K** using demographic and employment-related features from the UCI Adult (Census Income) dataset.

> **Note:** This project is for educational purposes. The dataset reflects historical census patterns and may contain social biases. Predictions should not be used to make real-world decisions about individuals.

## Project Overview

This notebook demonstrates a complete classification workflow:

- Download and load the dataset
- Inspect the data structure and summary statistics
- Identify missing values and duplicate rows
- Clean the dataset
- Perform exploratory data analysis (EDA)
- Separate features and target
- Split data into training and testing sets
- Preprocess numerical and categorical features
- Train and compare four classification algorithms
- Evaluate models using multiple metrics
- Visualize confusion matrices and ROC curves
- Inspect feature importance for tree-based models
- Save the model comparison results to CSV

## Dataset

- **Name:** UCI Adult / Census Income dataset
- **Source:** [UCI Machine Learning Repository — Adult](https://archive.ics.uci.edu/dataset/2/adult)
- **Target column:** `income`
- **Task:** Binary classification

The target is mapped to numeric labels:

| Original label | Numeric label |
|---|---:|
| `<=50K` | `0` |
| `>50K` | `1` |

Features include age, workclass, education, marital status, occupation, relationship, race, sex, hours worked per week, capital gain/loss, and native country.

The notebook downloads the dataset when run, so an internet connection is required for the initial download.

## Algorithms Used

The notebook compares these four supervised classification algorithms:

1. **Logistic Regression** — a linear classification baseline.
2. **Decision Tree Classifier** — learns decision rules by splitting feature values.
3. **Random Forest Classifier** — combines predictions from multiple decision trees.
4. **Gradient Boosting Classifier** — builds trees sequentially, with each stage improving the ensemble.

The notebook evaluates all four models on the same held-out test set.

## Preprocessing

Preprocessing is included in scikit-learn pipelines to help prevent data leakage.

- **Missing-value inspection:** counts and percentages are reported for each column.
- **Duplicate handling:** exact duplicate rows are removed.
- **Numerical features:** missing values are imputed with the median, then features are standardized using `StandardScaler`.
- **Categorical features:** missing values are imputed with the most frequent value, then encoded using `OneHotEncoder`.
- **Train/test split:** an 80/20 stratified split preserves approximately the same class proportions in each set.
- **Unknown categories:** `OneHotEncoder(handle_unknown="ignore")` handles categories that appear in test data but not training data.

## Evaluation Metrics

The notebook reports:

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** among predicted `>50K` cases, the proportion that are actually `>50K`.
- **Recall:** among actual `>50K` cases, the proportion correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** measures how well predicted probabilities distinguish the two classes.

It also plots:

- Model metric comparison
- Confusion matrices
- ROC curves
- Feature importance for Decision Tree, Random Forest, and Gradient Boosting

The notebook generates the scores when it runs; no performance values are hard-coded in this README.

## Project Structure

```text
adult-income-prediction/
├── adult_income_prediction_four_algorithms.ipynb
├── README.md
└── adult_income_model_comparison.csv   # generated after running the notebook
```

The CSV file is created by the final notebook cell.

## Requirements

- Python 3.9 or later recommended
- Jupyter Notebook or JupyterLab
- Internet connection to download the dataset

Python libraries used:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- requests

## How to Run

### 1. Clone your repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your GitHub repository URL and folder name.

### 2. Open the notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open `adult_income_prediction_four_algorithms.ipynb`.

### 3. Run the cells

Run the notebook from top to bottom. The first code cell installs the required packages, and the dataset-loading cell downloads the data.

If your environment does not support `%pip`, install the packages from a terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

### 4. Review the outputs

After the notebook finishes, review the comparison table and visualizations. The notebook saves the comparison table as:

```text
adult_income_model_comparison.csv
```

## Key Learning Outcomes

- Understanding supervised binary classification
- Performing data inspection and exploratory analysis
- Handling missing values and duplicate records
- Encoding categorical data and scaling numerical features
- Building leakage-aware preprocessing pipelines
- Comparing multiple machine learning algorithms
- Interpreting classification metrics and visualizations
- Exporting model evaluation results

## Limitations

- The dataset is historical and may not represent current populations or economic conditions.
- Demographic data can encode historical inequities and bias.
- Model performance alone does not establish fairness, causation, or suitability for real-world use.
- The notebook's highest test F1 score is descriptive. For a formal model-selection workflow, use validation data to select and tune models, then evaluate the chosen model on a final untouched test set.

## License and Attribution

The dataset is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). Check the source page for dataset details and applicable terms. Add a license file to this repository if you intend to publish the code under a specific software license.
