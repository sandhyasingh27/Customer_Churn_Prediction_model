# Customer_Churn_Prediction_model
Customer churn is one of the biggest challenges faced by subscription-based businesses. In this project, I built a predictive machine learning model that identifies customers who are likely to churn, using a telecom customer dataset. The goal is to help the business retain valuable customers by understanding key churn drivers.

# 📊 Customer Churn Prediction using Machine Learning



## Project Overview

Customer churn refers to when customers stop doing business with a company.  

This project uses **machine learning and data analytics** to predict which telecom customers are most likely to churn.  

By identifying these customers early, a business can take proactive steps to retain them.



---



## Objectives

- Explore and clean telecom customer data.  

- Identify key factors contributing to customer churn.  

- Build a machine learning model to predict churn.  

- Evaluate model performance using multiple metrics.  

- Use **SHAP** (Explainable AI) to interpret model predictions.



---



## Tools & Libraries

| Library | Purpose |

|----------|----------|

| **Pandas, NumPy** | Data manipulation \& analysis |

| **Matplotlib, Seaborn, Plotly** | Data visualization |

| **Scikit-learn** | Model training, evaluation, and cross-validation |

| **SHAP** | Feature importance and interpretability |

| **Jupyter Notebook** | Interactive development environment |



---



## 📂 Dataset Description

**Dataset:** Telco Customer Churn  



| Column | Description |

|---------|-------------|

| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demographic information |

| `tenure` | Number of months the customer has stayed |

| `PhoneService`, `InternetService`, `Contract`, etc. | Service-related information |

| `MonthlyCharges`, `TotalCharges` | Billing and payment information |

| `Churn` | Target variable (Yes/No) |

| `Churn_numeric` | Converted target (1 = Churn, 0 = Not churn) |



---



## ⚙️ Project Workflow



### **1️⃣ Data Preprocessing**

\- Dropped irrelevant columns (`customerID`).  

\- Converted `Churn` column into numerical format → `Churn\_numeric`.  

\- Encoded categorical columns using `pd.get\_dummies()`.  

\- Checked for missing values and data types.



```python

df\_model = df.drop("Churn", axis=1)

df\_encoded = pd.get\_dummies(df\_model, drop\_first=True)
```

### **2️⃣ Define Features and Target
```python

X = df\_encoded.drop('Churn\_numeric', axis=1)

y = df\_encoded\['Churn\_numeric']
```


### **3️⃣ Model Building and Validation**



Used Logistic Regression and cross-validation strategies to ensure robust evaluation.

```python

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score



model = LogisticRegression(max_iter=1000, solver='liblinear')



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')



rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

scores = cross_val_score(model, X, y, cv=rskf, scoring='accuracy')
```


### **4️⃣ Model Evaluation**
```python
from sklearn.metrics import (

&nbsp;   accuracy_score, precision_score, recall_score,

&nbsp;   f1_score, roc_auc_score, classification_report, confusion_matrix

)



y_pred = model.predict(X_test)



print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

```



Results:



Metric	Score

Accuracy	0.8027

Precision	0.6529

Recall	0.5481

F1 Score	0.5959

ROC-AUC	0.8423



Confusion Matrix:



[[926 109]

&nbsp;[169 205]]


Notes:
The model correctly predicts ~80% of customers overall.
ROC-AUC of 0.84 indicates strong discrimination ability.
Recall (0.55) shows moderate ability to identify true churners.

## **5️⃣ SHAP Explainability**
```python
import shap



explainer = shap.Explainer(model, X_train)

shap_values = explainer(X_test)



shap.summary_plot(shap_values, X_test)
```




Insights:



Higher MonthlyCharges increase churn probability.



Shorter Tenure correlates with higher churn.



Customers without internet services tend to churn less.



### **📈 Key Takeaways**



Shorter-tenure, high-paying customers are at greater risk.



Logistic Regression provides interpretable insights.



SHAP enhances transparency by showing how each feature affects predictions.



Business teams can use these insights for targeted retention campaigns.

💼 Author
Sandhya Singh

Data Analyst | Python | SQL | Power BI | Machine Learning

LinkedIn - https://www.linkedin.com/in/sandhya-singh-b6b4b519b/
