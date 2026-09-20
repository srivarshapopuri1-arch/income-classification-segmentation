# Project Study Guide

This document explains the project in enough detail to review before an interview. It describes the implementation currently in the repository and separates design intent from results that still need to be regenerated.

## 1. Project in one sentence

The project uses Census Income (KDD) data for two related tasks: supervised income classification with a Random Forest and exploratory customer segmentation with K-Means.

## 2. What problem is being solved?

### Income classification

The supervised task converts the source income label into a binary target called `income_label`:

- `1`: the source label contains `50000+.`
- `0`: otherwise

The model then predicts this binary target from the remaining census attributes.

### Customer segmentation

The unsupervised task groups records into five clusters using selected demographic and income-related features. The clusters are profiled using average age, capital gains, weeks worked, common occupation, and percentage of higher-income records.

These are two separate analyses on the same source data. Classification predicts a known label; clustering looks for groups without a cluster label supplied in advance.

## 3. Dataset and project inputs

The notebook expects two files under `data/`:

```text
data/
├── census-bureau.data
└── census-bureau.columns
```

The dataset itself is intentionally not committed to the repository.

The notebook reads the column names first, then loads the data with those names. `?` values are treated as missing values.

## 4. Classification workflow

### Step 1: Define the target

The original text label is converted to `income_label`. The original `label`, the new target, and `year` are removed from the feature matrix.

### Step 2: Remove duplicate feature rows

Exact duplicate observations are removed using the feature values together with the target. This avoids accidentally discarding records that have identical features but different labels. The feature matrix, target, and working dataframe remain aligned.

### Step 3: Stratified train/test split

The data is split 80/20 with:

- `random_state=42`
- `stratify=y`

Stratification helps preserve the target-class proportions in both partitions.

A key technical decision is that the split happens **before learned preprocessing**. This prevents information from the test set from influencing imputation, encoding, outlier bounds, or feature selection.

### Step 4: Missing values and categorical encoding

Numeric columns use median imputation.

Categorical columns use most-frequent imputation followed by `OrdinalEncoder`. Unknown categories encountered outside the training data are encoded as `-1`.

The `ColumnTransformer` is fitted on `X_train_raw` and only transformed on `X_test_raw`.

### Step 5: Numeric outlier clipping

For each numeric feature, Q1, Q3, and IQR are calculated from the training data. The lower and upper clipping bounds are:

```text
lower = Q1 - 1.5 × IQR
upper = Q3 + 1.5 × IQR
```

Those training-derived bounds are then applied to both training and test data.

### Step 6: Feature selection

Each processed training feature is correlated with `y_train`. Features with absolute correlation greater than `0.01` are retained.

This is a simple filter-based feature-selection method. It remains intentionally close to the original project rather than introducing a more complex selection system.

Most importantly, selection is learned from training data only.

### Step 7: Random Forest

The classifier uses:

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
```

Why Random Forest is reasonable here:

- it handles nonlinear relationships and feature interactions;
- it works well as a strong baseline for tabular data;
- it does not require feature scaling;
- `class_weight="balanced"` helps address unequal class frequencies;
- feature importances provide a useful, though limited, interpretation tool.

### Step 8: Evaluation

The standard predictions are evaluated with accuracy and a classification report.

The notebook also calculates:

- precision;
- recall;
- F1-score;
- confusion matrix for the threshold analysis;
- ROC-AUC;
- average precision;
- training versus test accuracy;
- Random Forest feature importance.

ROC-AUC evaluates ranking ability across thresholds. Average precision is particularly useful when the positive class is less common because it summarizes the precision-recall relationship.

### Step 9: Probability threshold

The Random Forest produces probabilities for the positive class. The notebook separately evaluates:

```text
probability >= 0.7 → positive
probability < 0.7  → negative
```

The important interview point is that **0.7 is an analysis threshold, not a proven optimal threshold**.

Increasing a threshold generally makes positive predictions more selective. That can increase precision while reducing recall, but the actual trade-off must be measured on the data.

### Step 10: Model persistence

The saved classification artifact contains:

- fitted preprocessing transformer;
- training-derived numeric outlier bounds;
- fitted Random Forest;
- selected feature names;
- probability threshold.

Saving preprocessing metadata with the model is important because future observations need the same imputation, encoding, clipping bounds, and feature selection used during training.

## 5. Customer segmentation workflow

### Selected features

The K-Means analysis uses:

- age;
- education;
- marital status;
- major occupation code;
- sex;
- weeks worked in year;
- capital gains;
- income label.

### Encoding and scaling

Categorical segmentation features are ordinal-encoded. Then all selected clustering features are standardized with `StandardScaler`.

Scaling is important for K-Means because the algorithm is distance-based. Without scaling, variables with larger numerical ranges can dominate Euclidean distance.

### Elbow diagnostic

K-Means models are fitted for `k=1` through `k=10`. Their within-cluster sum of squares (WCSS/inertia) is plotted.

The project retains the original five-cluster solution. The elbow plot is a diagnostic supporting inspection of that choice; it should not be described as mathematical proof that five is the uniquely correct number of clusters.

### Five-cluster model

The final K-Means configuration is:

```python
KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10,
)
```

`k-means++` improves centroid initialization. `n_init=10` runs multiple initializations, and the fixed random seed improves reproducibility.

### Cluster profiling

After assigning a segment to every record, the notebook creates a profile table containing:

- average age;
- average capital gains;
- average weeks worked;
- most common occupation;
- percentage of records with the higher-income label.

It also creates a segment-size chart.

### Saved clustering artifact

The clustering artifact contains:

- categorical encoder;
- categorical feature names;
- StandardScaler;
- fitted K-Means model;
- feature schema.

This makes the preprocessing needed by the clustering model explicit instead of saving only transformed training data.

## 6. Outputs

When the notebook is run with the source data, it writes:

```text
models/
├── census_income_final_model.pkl
└── customer_segmentation_model.pkl

outputs/
└── customer_segmentation_results.csv

images/
├── elbow_method.png
└── segment_sizes.png
```

The repository also retains the original `Project Report.pdf` and `segmentation_output.png`.

## 7. What was improved during portfolio cleanup?

The cleanup deliberately preserved the project's purpose and core algorithms.

The important changes were:

1. Removed machine-specific Windows paths and switched to project-relative paths.
2. Added clear checks for missing source files.
3. Made duplicate removal operate on complete observations so conflicting labels are not silently discarded.
4. Split train/test data before fitting learned classification preprocessing.
5. Fit imputation and categorical encoding on training data only.
6. Added handling for unseen categorical values.
7. Learned outlier bounds from training data only and saved those bounds with the classifier artifact.
8. Performed correlation feature selection using training data only.
9. Added ROC-AUC and average precision to the evaluation code.
10. Clarified the meaning of the 0.7 threshold.
11. Saved fitted preprocessing components with the models.
12. Made K-Means reproducible with fixed initialization settings.
13. Removed the stale root-level generated model artifact.
14. Cleared stale notebook outputs after changing the methodology.
15. Improved repository paths, requirements, Git ignores, and README documentation.

## 8. What you should not claim

Do not quote new accuracy, ROC-AUC, average-precision, precision, recall, F1, or clustering results until the corrected notebook has been rerun using the source dataset.

Do not say the 0.7 threshold was mathematically optimized. The current repository treats it as an analytical threshold.

Do not say five clusters are definitively optimal. The project uses an elbow diagnostic and retains the original five-cluster design.

Do not describe the segmentation as completely independent of income: `income_label` is one of its clustering features.

Do not describe this as a production ML system. It is an individual notebook-based machine-learning analysis and portfolio project.

## 9. Limitations you can discuss

### Ordinal encoding

Categorical values receive numeric codes. Those codes should not be interpreted as meaningful category order. This approach is retained as a practical baseline close to the original project.

### Correlation-based feature selection

The filter considers individual feature-target correlation and does not capture every nonlinear or interaction effect that a Random Forest may exploit.

### Threshold selection

The 0.7 threshold is illustrative. A real deployment would select a threshold according to business costs, validation results, and the desired precision/recall balance.

### Segmentation target dependence

Because `income_label` participates in K-Means, the clusters are intentionally income-oriented. Excluding it would answer a different question: whether demographic/behavioral variables naturally produce segments that can later be compared by income.

### External dataset

The source data is not committed, so a fresh clone is not fully executable until the two source files are added to `data/`.

## 10. Interview explanation

A concise explanation:

> I built a two-part machine-learning analysis using Census Income data. The supervised portion predicts the income class using a class-weighted Random Forest. I use a stratified train/test split and fit preprocessing and feature selection only on the training set to avoid leakage. I evaluate the model with classification metrics, ROC-AUC and average precision, and I also examine how a stricter 0.7 probability threshold changes classification behavior. The second portion uses K-Means to create five demographic and income-oriented segments. I encode categorical variables, standardize the clustering features, inspect an elbow plot, and profile the resulting clusters. I also persist the fitted preprocessing objects with both models so the transformations are reproducible.

## 11. Questions you should be ready to answer

**Why split before preprocessing?**  
To prevent information from the test set from influencing transformations or feature selection. The test set should approximate unseen data.

**Why stratify the split?**  
The target classes are unequal, so stratification keeps approximately the same class proportions in training and test sets.

**Why use class weights?**  
They make mistakes on the less common class carry more influence during model fitting instead of allowing the majority class to dominate solely because it has more observations.

**Why Random Forest?**  
It is a strong tabular baseline that captures nonlinear patterns and interactions, requires relatively little feature transformation, and supports class weighting and feature importance.

**Why not scale data for Random Forest?**  
Tree splits depend on ordering and thresholds rather than Euclidean distance, so scaling is generally unnecessary for the classifier.

**Why scale for K-Means?**  
K-Means uses distances. Standardization prevents features with large numerical scales from dominating the clustering objective.

**What does ROC-AUC mean?**  
It summarizes how well predicted scores rank positive examples above negative examples across classification thresholds.

**Why average precision?**  
It summarizes precision-recall performance and is informative when the positive class is comparatively uncommon.

**What happens when the probability threshold rises?**  
The classifier requires more confidence before assigning the positive class. The typical trade-off is fewer positive predictions, potentially higher precision, and potentially lower recall.

**Why use the elbow method?**  
It compares how WCSS decreases as more clusters are added. The bend in the curve can provide a practical clue for choosing a cluster count, but it is not definitive proof.

**What would you improve next?**  
First rerun and validate the corrected pipeline on the source data. Then compare alternative categorical representations or models, use cross-validation for model/threshold decisions, add additional clustering-quality diagnostics, and consider a Scikit-learn pipeline that packages more of the classification transformations into one fitted object.

## 12. The five concepts to remember

If you have limited preparation time, understand these especially well:

1. **Data leakage:** test data must not influence learned preprocessing or feature selection.
2. **Class imbalance:** accuracy alone can be misleading; precision, recall, F1, ROC-AUC, and average precision provide more context.
3. **Probability thresholds:** a classifier's probability can be converted to a class using different thresholds depending on the desired trade-off.
4. **Random Forest vs. K-Means:** Random Forest is supervised and predicts a known target; K-Means is unsupervised and creates groups based on distance.
5. **Scaling:** Random Forest generally does not require it, while K-Means does because distance drives cluster assignment.

---

This guide documents the current repository implementation. Performance values should be added only after the corrected notebook is rerun against the source data.
