
# README
## Income Classification & Customer Segmentation Project

Prepared by: Sri Varsha Popuri

---

## 1. Project Description

This project contains two machine learning implementations:

1. Income Classification Model (Supervised Learning)
   - Predicts whether an individual earns more than $50K annually.
   - Built using a Random Forest Classifier.
   - Includes probability threshold tuning for business optimization.

2. Customer Segmentation Model (Unsupervised Learning)
   - Groups customers into 5 distinct personas using K-Means clustering.
   - Produces segment-level business insights and marketing strategies.

The complete implementation is provided in:
classification_segmentation.ipynb

---

## 2. Files Included

- classification_segmentation.ipynb – Full implementation (training + evaluation)
- census_income_final_model.pkl – Saved classification model
- customer_segmentation_model.pkl – Saved clustering model
- Final_Customer_Segmentation_Results.csv – Final segmentation output
- Project Report.pdf – Technical documentation
- README.md – Execution instructions

---

## 3. System Requirements

Python Version:
Python 3.9 or higher (3.10 recommended)

Required Libraries:
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
jupyter

Install dependencies using:

pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter

---

## 4. Environment Setup (Compilation Step)

Since this project is written in Python, no traditional compilation is required.
However, environment setup is required before execution.

Step 1: Create Virtual Environment (Recommended)

python -m venv venv

Activate environment:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

Step 2: Install Dependencies

pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter

---

## 5. Execution Instructions

Step 1: Navigate to Project Folder

cd <project_folder_path>

Step 2: Launch Jupyter Notebook

jupyter notebook

Open:
classification_segmentation.ipynb

Step 3: Run All Cells

Kernel → Restart & Run All

This will:
- Load the dataset (census-bureau.data)
- Perform preprocessing
- Train Random Forest classification model
- Evaluate performance metrics
- Apply probability threshold tuning (0.7)
- Train K-Means segmentation model
- Generate Elbow Method visualization
- Assign customer clusters
- Save model artifacts and final outputs

---

## 6. Expected Outputs

After successful execution:

Classification:
- census_income_final_model.pkl
- Model evaluation metrics displayed in notebook

Segmentation:
- customer_segmentation_model.pkl
- Final_Customer_Segmentation_Results.csv
- Cluster profiling summary table

---

## 7. Reproducing Results

To reproduce results:

1. Ensure census-bureau.data is in the same directory as the notebook.
2. Run all notebook cells from top to bottom.
3. Confirm output files are generated in the project directory.

---

## 8. Notes

- If a scikit-learn version mismatch warning appears when loading .pkl files,
  retrain the models in your current environment.
- All preprocessing steps are included within the notebook.
- Feature scaling (StandardScaler) is applied before clustering to ensure balanced distance calculations.

---

End of README
