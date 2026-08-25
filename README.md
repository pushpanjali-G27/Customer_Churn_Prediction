# Telco Customer Churn Prediction

## Project Overview

This project uses Machine Learning to predict whether a telecom customer is likely to churn.

The project follows an end-to-end Machine Learning workflow, starting from data exploration and cleaning to model training, evaluation, comparison, and saving the final model.

---

## Business Problem

Customer churn is a major challenge for telecom companies.

The goal of this project is to identify customers who are likely to leave the company so that businesses can take proactive actions to improve customer retention.

---

## What is Customer Churn?

Customer churn occurs when a customer stops using a company's service.

In this dataset:

* `Yes` → Customer churned
* `No` → Customer did not churn

---

## Dataset

The project uses the **Telco Customer Churn dataset**.

The dataset contains customer information such as:

* Tenure
* Contract type
* Internet service
* Payment method
* Monthly charges
* Total charges
* Customer services
* Demographic information
* Churn status

### Target Variable

`Churn`

* `Yes` → Churn
* `No` → No Churn

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

---

# Project Workflow

## 1. Data Exploration

The dataset was explored to understand:

* Number of rows and columns
* Column names
* Data types
* Unique values
* Missing values
* Empty values
* Duplicate records
* Target variable distribution

---

## 2. Data Cleaning

The following cleaning steps were performed:

* Checked for missing values
* Checked for empty values
* Checked data types
* Converted required columns to appropriate data types
* Removed unnecessary customer identifiers
* Prepared the target variable

---

## 3. Exploratory Data Analysis (EDA)

Different customer features were analyzed against churn.

### Key Findings

* Customers with shorter tenure showed higher churn.
* Month-to-month contract customers had higher churn.
* Long-term contracts were associated with lower churn.
* Churn behavior differed across internet service types.
* Customer charges were analyzed to understand their relationship with churn.

---

## 4. Feature and Target Preparation

The target variable was separated from the input features.

**Target Variable:**

`Churn`

The dataset was divided into:

* 80% → Training Data
* 20% → Testing Data

A stratified split was used to maintain the churn distribution.

---

## 5. Data Preprocessing

The dataset contains both numerical and categorical features.

### Numerical Features

`StandardScaler` was used to scale numerical features.

### Categorical Features

`OneHotEncoder` was used to convert categorical features into numerical values.

A `ColumnTransformer` and Scikit-learn `Pipeline` were used to combine the preprocessing and model steps.

---

## 6. Logistic Regression

Logistic Regression was trained as the baseline Machine Learning model.

### Results

| Metric    | Score |
| --------- | ----: |
| Accuracy  |   81% |
| Precision |   66% |
| Recall    |   56% |
| F1-Score  |   60% |

---

## 7. Random Forest

Random Forest was trained as another classification model and compared with Logistic Regression.

### Results

| Metric    | Score |
| --------- | ----: |
| Accuracy  |   78% |
| Precision |   62% |
| Recall    |   48% |
| F1-Score  |   54% |

---

## 8. Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |      81% |       66% |    56% |      60% |
| Random Forest       |      78% |       62% |    48% |      54% |

Logistic Regression performed better than Random Forest on this dataset based on the evaluated metrics.

---

## 9. Class Imbalance

The dataset contains more non-churn customers than churn customers.

To improve the detection of churners, `class_weight="balanced"` was used with Logistic Regression.

This gives more importance to the minority churn class.

---

## 10. Balanced Logistic Regression

The balanced model achieved:

| Metric    | Score |
| --------- | ----: |
| Accuracy  |   74% |
| Precision |   50% |
| Recall    |   78% |
| F1-Score  |   61% |

The recall improved from:

`56% → 78%`

This means the balanced model was able to identify more customers who actually churned.

---

## 11. Final Model Comparison

| Model                        | Accuracy | Precision | Recall | F1-Score |
| ---------------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression          |      81% |       66% |    56% |      60% |
| Random Forest                |      78% |       62% |    48% |      54% |
| Balanced Logistic Regression |      74% |       50% |    78% |      61% |

### Final Model

**Balanced Logistic Regression** was selected as the final model because identifying potential churners is the main objective, making recall an important metric.

---

## 12. Model Interpretation

Logistic Regression coefficients were analyzed to understand which features influenced churn predictions.

Important observations included:

* Higher tenure was associated with lower churn likelihood.
* Two-year contracts were associated with lower churn likelihood.
* Month-to-month contracts were associated with higher churn likelihood.
* Fiber Optic internet service showed a positive association with churn.

These are model associations and should not be interpreted as direct causes of churn.

---

## 13. Save the Final Model

The final model was saved using Joblib:

```python
import joblib

joblib.dump(
    balanced_logistic_model,
    "telco_churn_model.pkl"
)
```

The saved model contains the preprocessing steps and trained model, allowing it to be reused for future predictions.

---

## Key Business Insights

* New customers are more likely to churn.
* Month-to-month customers are a higher-risk group.
* Longer-term contracts are associated with lower churn.
* Machine Learning can help identify potential churners.
* Churn predictions can support targeted customer retention strategies.

---

## Project Structure

```text
Telco-Customer-Churn/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── Telco_Customer_Churn.ipynb
├── telco_churn_model.pkl
└── README.md
```

---

## Conclusion

This project demonstrates an end-to-end customer churn prediction workflow using Python and Machine Learning.

The project covers:

* Data exploration
* Data cleaning
* Exploratory Data Analysis
* Feature and target preparation
* Data preprocessing
* Model training
* Model evaluation
* Model comparison
* Class imbalance handling
* Model interpretation
* Model saving

The final model, **Balanced Logistic Regression**, achieved a **78% recall**, making it useful for identifying customers who are likely to churn.


