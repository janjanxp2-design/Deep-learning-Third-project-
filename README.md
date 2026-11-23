# Deep-learning-Third-project-
[README_adult.md](https://github.com/user-attachments/files/35273100/README_adult.md)
# Adult Income Prediction Using Deep Learning

## Project Overview

This project uses the Adult Census Income dataset to build a binary classification model that predicts whether a person's annual income is:

- `<=50K`
- `>50K`

The notebook (`adult.ipynb`) follows a complete machine-learning workflow: data inspection, cleaning, exploratory data analysis, preprocessing, class balancing, neural-network training, evaluation, threshold tuning, feature-importance analysis, and saving the trained model and preprocessing object.

## Dataset

The notebook loads the CSV file from this Google Drive path:

```text
/content/drive/MyDrive/Data of Dl/adult/adult.csv
```

Before running the notebook, place `adult.csv` at this location in Google Drive, or update the `pd.read_csv()` path to match your own dataset location.

The target column is `income`, with the labels mapped as follows:

| Original label | Encoded value |
|---|---:|
| `<=50K` | 0 |
| `>50K` | 1 |

The notebook drops the `fnlwgt` and `education` columns. It handles `?` entries in `workclass`, `occupation`, and `native.country` as missing values, then fills those missing values with an `Unknown` category (the notebook uses `Unknwn` for `native.country`).

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- imbalanced-learn (SMOTE)
- Joblib

## Installation

Install the libraries used by the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow imbalanced-learn joblib
```

The notebook is written with a Google Colab-style Google Drive file path. If you run it in Colab, mount Google Drive before loading the dataset and before saving the output files.

## Workflow

1. **Import libraries and set random seeds** – Uses seed `42` for Python, NumPy, and TensorFlow.
2. **Load and inspect data** – Reads the CSV and checks its structure and missing values.
3. **Exploratory data analysis** – Examines the income-class distribution, selected numeric-feature distributions by income, and correlations among selected numeric features.
4. **Clean data** – Drops `fnlwgt` and `education`, converts `?` markers to missing values in selected categorical columns, and fills missing values.
5. **Prepare features and target** – Separates `income` from the input features and identifies numeric and categorical columns.
6. **Split the data** – Creates stratified training, validation, and test sets using a 70% / 15% / 15% split.
7. **Preprocess features** – Applies `StandardScaler` to numeric columns and `OneHotEncoder(handle_unknown='ignore')` to categorical columns. The preprocessor is fitted on the training data only.
8. **Handle class imbalance** – Applies SMOTE to the training set only.
9. **Build and train a neural network** – Uses a Keras sequential binary-classification model with dense layers, ReLU activations, batch normalization, and dropout.
10. **Train with callbacks** – Uses early stopping, learning-rate reduction on plateau, and model checkpointing.
11. **Evaluate the model** – Reports classification metrics and ROC-AUC on the held-out test set, and plots a confusion matrix, ROC curve, and precision-recall curve.
12. **Tune the decision threshold** – Searches thresholds from 0.10 to 0.89 to find the threshold with the best F1 score on the test predictions.
13. **Analyze feature importance** – Includes a permutation-importance function based on the drop in ROC-AUC.
14. **Save and use the model** – Saves the Keras model and fitted preprocessing object, and defines a function for predicting income from raw person records.

## Output Files

The notebook saves these files to Google Drive:

```text
/content/drive/MyDrive/Data of Dl/adult/adult_model.keras
/content/drive/MyDrive/Data of Dl/adult/adult_preprocessor.pkl
```

It also configures a best-model checkpoint named `best_model.keras` during training, in the notebook's current working directory.

## Making Predictions

The notebook defines `predict_income(raw_person_df, model, preprocessor, threshold=best_threshold)`. It transforms raw input rows with the fitted preprocessor and returns the input data with two additional columns:

- `high_income_probability` – predicted probability of income above 50K.
- `predicted_income` – predicted class (`>50K` or `<=50K`).

A sample prediction is demonstrated using the first five rows of the test feature set.

## Running the Notebook

1. Open `adult.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required dependencies.
3. Make the Adult dataset CSV available at the path expected by the notebook.
4. If using Google Colab, mount Google Drive.
5. Run the cells in order, from top to bottom.
6. Review the plots, evaluation reports, and saved model files.

## Notes

- The notebook's dataset path and save paths are specific to Google Colab / Google Drive. Update them when running in another environment.
- The notebook searches for a threshold that maximizes F1 using the test-set predictions. For an unbiased final performance estimate, threshold selection should be performed on the validation set, with the test set kept for final evaluation.
- The README describes the workflow implemented in the notebook; it does not claim a particular accuracy or other performance result.

## Project Status

The notebook contains the data-preparation, model-training, evaluation, model-saving, and sample-prediction workflow.

## Author

Add your name here.
