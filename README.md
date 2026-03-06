# Customer Churn Prediction System (Machine Learning)

## 🎯 Problem Statement
Telecom companies lose revenue when customers cancel subscriptions (churn).
The goal of this project is to predict which customers are likely to churn so the company can intervene early with retention strategies.

---

## 📊 Dataset
**Dataset:** Telco Customer Churn  

- 7043 customers  
- 20 features + churn target  

Example features:
- tenure
- MonthlyCharges
- Contract type
- InternetService
- PaymentMethod
- TechSupport
- Streaming services

**Target variable:**  
Churn (Yes / No)

Baseline churn rate: **~27%**

---

## 🧹 Data Preparation
Steps performed:

- Removed identifier column (`customerID`)
- Converted `TotalCharges` to numeric
- Handled missing values (11 rows)

Preprocessing pipeline:

- Median imputation for numeric features
- Most frequent imputation for categorical features
- One-hot encoding
- Standard scaling

All preprocessing is handled inside a **scikit-learn pipeline** to prevent data leakage.

---

## 🤖 Models Tested

| Model | ROC-AUC | Recall |
|------|------|------|
| Logistic Regression | ~0.84 | 0.53 |
| Random Forest | ~0.82 | 0.48 |
| Gradient Boosting | ~0.83 | 0.52 |

Handling class imbalance:

`class_weight="balanced"`

**Final Model:** Balanced Logistic Regression

Performance:

- ROC-AUC: ~0.84
- Recall: ~0.79–0.80
- Precision: ~0.51
- Accuracy: ~0.74

The model prioritizes **recall to detect more churners**.

---

## 🔍 Key Drivers of Churn

**Increase churn risk**
- Month-to-month contracts
- Fiber optic internet
- Streaming services
- Short tenure

**Reduce churn risk**
- Longer tenure
- Two-year contracts
- DSL internet service

These results align with real-world telecom behavior.

---

## 💼 Business Strategy

Instead of using a fixed threshold, customers are ranked by churn probability.

Deployment strategy:

1. Score all customers weekly
2. Rank by predicted churn probability
3. Target **top 25% highest-risk customers**

Results:

- Baseline churn rate: **27%**
- Top 25% risk group churn rate: **62.5%**

This is **more than 2× improvement in targeting efficiency**.

Retention actions may include:

- Discounts
- Contract upgrades
- Loyalty programs
- Customer service outreach

---

## ⚙️ Deployment Simulation

The trained model is saved using:

```python
import joblib
joblib.dump(model_clean, "churn_model.pkl")
