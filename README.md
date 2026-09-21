# Income Classification & Customer Segmentation

This project explores two machine-learning tasks using the Census Income (KDD) data:

- **Income classification:** predict whether a record belongs to the higher-income class using a Random Forest classifier.
- **Customer segmentation:** group records into five descriptive segments using K-Means clustering.

The project remains notebook-based so the full analysis can be followed from data loading through modeling and interpretation.

## Methodology

### Income classification

The workflow removes duplicate feature rows, creates a stratified train/test split, and fits preprocessing only on the training data. Numeric missing values are imputed with the training median. Categorical values are imputed and ordinal-encoded with explicit handling for categories that appear only in the test set.

Numeric outlier bounds and correlation-based feature selection are also learned from the training partition. A class-weighted Random Forest is then evaluated with classification metrics, ROC-AUC, average precision, a confusion matrix, and a separate 0.7 probability-threshold analysis.

The threshold analysis is included to show the precision/recall trade-off; it is not presented as a universally optimal business threshold.

### Customer segmentation

The segmentation analysis uses demographic and income-related variables from the original project. Categorical values are encoded, all selected features are standardized, and K-Means is fitted with a fixed random seed and explicit `n_init`. The notebook includes an elbow diagnostic, cluster-size visualization, and a segment profile table.

## Repository structure

```text
.
├── classification_segmentation.ipynb
├── Project Report.pdf
├── segmentation_output.png
├── requirements.txt
├── .gitignore
├── images/
├── models/
└── outputs/
```

`Project Report.pdf` and `segmentation_output.png` are retained from the original project. Generated models, CSV outputs, and notebook-generated plots are not committed; rerunning the notebook creates them locally.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Data

The source dataset is not committed to this repository. To run the notebook, create a `data/` directory and add:

```text
data/
├── census-bureau.data
└── census-bureau.columns
```

The notebook checks for both files and raises a clear error if either is missing.

## Run the analysis

```bash
jupyter notebook classification_segmentation.ipynb
```

Run the notebook from top to bottom. It creates local output directories as needed and writes:

- classification and clustering artifacts to `models/`
- segmentation results to `outputs/`
- generated plots to `images/`

## Reproducibility

Random seeds are fixed for the train/test split, Random Forest, and K-Means steps. Preprocessing and feature selection for classification are fitted on training data only to avoid using test-set information during model development.

Notebook outputs were cleared after the workflow was corrected because the source dataset is not committed and the revised pipeline could not be rerun from the repository alone. No new performance numbers are claimed until the updated workflow is executed against the source data.

## Limitations

The classification workflow uses ordinal encoding for categorical variables because it stays close to the original implementation. For a tree-based model this is a practical baseline, but the assigned category numbers should not be interpreted as meaningful order.

The clustering analysis is exploratory. Including `income_label` among the clustering features makes the resulting segments useful for income-oriented profiling, but it also means the clusters are not independent of the classification target.

The repository intentionally remains a compact notebook project rather than being expanded into a production application.
