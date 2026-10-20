# Bank Customer Churn Prediction

## Project Overview

This project focuses on predicting whether a bank customer is likely to leave the bank using Machine Learning. It uses customer information such as credit score, age, account balance, geography, and activity status to identify potential churn.

The project explores Support Vector Machine (SVM), class imbalance handling, and hyperparameter tuning to improve churn detection.

## Objectives

- Predict whether a customer will leave the bank.
- Analyze customer data to understand churn patterns.
- Handle imbalanced data using Random Under-Sampling and Random Over-Sampling.
- Train and evaluate an SVM classification model.
- Improve model performance using GridSearchCV.

## Dataset

- **Dataset:** Bank Churn Modelling
- **Source:** [YBI Foundation Dataset](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Bank%20Churn%20Modelling.csv)
- **Total Records:** 10,000 customers
- **Target Variable:** `Churn`
  - `0` — Customer stays
  - `1` — Customer leaves

### Features Used

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Zero Balance

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

## Project Workflow

1. **Data Loading:** Imported the dataset using Pandas.
2. **Data Exploration:** Examined dataset structure, data types, duplicate customer IDs, and churn distribution.
3. **Data Preprocessing:** Encoded categorical features and created a Zero Balance feature.
4. **Feature Selection:** Separated input features and the target variable.
5. **Handling Imbalanced Data:** Applied Random Under-Sampling and Random Over-Sampling.
6. **Train-Test Split:** Divided the data into training and testing sets.
7. **Feature Scaling:** Standardized numerical features using StandardScaler.
8. **Model Training:** Trained a Support Vector Machine (SVM) classifier.
9. **Hyperparameter Tuning:** Used GridSearchCV to find better model parameters.
10. **Model Evaluation:** Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrices.

## Model Performance

The following results are from the experiments recorded in the notebook.

| Model | Accuracy | Churn Recall | Churn F1-Score |
|---|---:|---:|---:|
| Baseline SVM | 84% | 28% | 41% |
| Tuned SVM | 79% | 37% | 41% |
| SVM with Random Under-Sampling | 76% | 72% | 75% |
| Tuned SVM with Random Under-Sampling | 75% | 74% | 75% |
| SVM with Random Over-Sampling | 51% | 0% | 0% |

## Key Findings

- The dataset contains an imbalance between customers who stay and customers who leave.
- The baseline SVM achieved 84% accuracy but detected only 28% of actual churn cases.
- Random Under-Sampling improved churn recall to 72%.
- Hyperparameter tuning on the undersampled data increased churn recall to 74%.
- The recorded Random Over-Sampling experiment failed to identify churn cases.
- Recall and F1-score are important metrics because missing customers who may leave can be costly for a bank.


## Future Improvements

- Improve preprocessing and prevent data leakage during feature scaling.
- Apply sampling only to the training data.
- Compare SVM with Logistic Regression, Random Forest, and XGBoost.
- Use stratified cross-validation for more reliable evaluation.
- Tune the classification threshold to balance precision and recall.
- Build a simple web application to predict customer churn.

## Conclusion

This project demonstrates a machine learning workflow for bank customer churn prediction. It highlights the importance of handling imbalanced datasets and evaluating models using recall and F1-score rather than relying on accuracy alone.


