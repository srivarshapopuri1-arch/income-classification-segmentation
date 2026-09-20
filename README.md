# Income Classification & Customer Segmentation

A machine learning portfolio project demonstrating both **supervised classification** and **unsupervised customer segmentation** on census-income data.

## Project Overview

This project solves two related machine learning problems:

1. **Income Classification** — predicts whether an individual earns more than $50K annually using a Random Forest classifier.
2. **Customer Segmentation** — groups records into five segments using K-Means clustering to support persona analysis and targeted decision-making.

The notebook covers data loading, preprocessing, exploratory analysis, feature preparation, model training, evaluation, probability-threshold tuning, clustering, cluster profiling, visualization, and model persistence.

## Machine Learning Workflow

### Income Classification
- Data cleaning and preprocessing
- Categorical feature encoding
- Random Forest classification
- Model evaluation
- Probability-threshold tuning for decision optimization

### Customer Segmentation
- Feature selection
- Standardization
- K-Means clustering
- Elbow-method analysis
- Five-cluster segmentation
- Cluster profiling and visualization

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## Repository Contents

| File | Purpose |
| --- | --- |
| `classification_segmentation.ipynb` | End-to-end analysis, model training, evaluation, and segmentation |
| `customer_segmentation_model.pkl` | Saved clustering artifact |
| `segmentation_output.png` | Customer-segmentation visualization |
| `Project Report.pdf` | Detailed project documentation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/srivarshapopuri1-arch/income-classification-segmentation.git
cd income-classification-segmentation
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the source dataset

The notebook was developed with the Census Income (KDD) dataset. The source data is not currently stored in this repository. Place the required data and column-definition files in a local `data/` directory before running the notebook.

The notebook uses portable project-relative paths. Place `census-bureau.data` and `census-bureau.columns` inside `data/`; generated models, outputs, and visualizations are written to dedicated project folders.

### 5. Run the notebook

```bash
jupyter notebook classification_segmentation.ipynb
```

Then run the cells from top to bottom.

## Outputs

The notebook demonstrates:
- Income-classification model training and evaluation
- Probability-threshold tuning
- Customer segmentation with K-Means
- Cluster profiling
- Segmentation visualization
- Model artifact generation

## Portfolio Context

This project demonstrates foundational machine learning skills across classification, clustering, preprocessing, model evaluation, and analytical interpretation. It complements production ML, MLOps, and Generative AI projects in an AI/ML engineering portfolio.

## Engineering Improvements Included

- Portable project-relative dataset paths
- Reproducible train/test split with stratification and fixed random seeds
- Dedicated `models/`, `outputs/`, and `images/` output locations created by the notebook
- Saved fitted `StandardScaler` with the clustering model rather than transformed training data
- Explicit K-Means `n_init` for reproducibility
- Dependency and Git ignore files for cleaner setup

## Future Enhancements

- Package preprocessing and models into reusable Scikit-learn pipelines
- Add automated tests and CI
- Add a lightweight inference API
- Add model-performance and cluster-quality metrics directly to this README
