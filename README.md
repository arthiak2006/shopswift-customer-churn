# Predictive Customer Churn Engine & Automated Retention Pipeline

## Project Overview
**Organization:** ShopSwift Retail Pvt. Ltd. (B2C E-Commerce)  
**Methodology:** CRISP-DM + Agile Scrum (8-Week Execution Plan)  
**Primary Objective:** Predict customer churn 30 days in advance with **ROC-AUC >= 0.86** and **Precision@Top10% >= 0.84** to reduce annual revenue leakage.

## Repository Structure
```text
shopswift-customer-churn-prediction/
├── README.md                  # Project architecture, KPIs & setup guide
├── requirements.txt           # Python dependencies
└── churn_pipeline.py          # End-to-end Scikit-Learn + XGBoost + SHAP pipeline
```

## Tech Stack
* **Data Processing:** Pandas, NumPy, SQLAlchemy, PySpark
* **Feature Engineering:** Scikit-Learn (ColumnTransformer, RobustScaler, Pipeline)
* **Modeling & Tuning:** XGBoost, LightGBM, Imbalanced-learn (SMOTETomek), Optuna
* **Explainability & Monitoring:** SHAP, Evidently AI
* **Deployment:** Docker, Apache Airflow, Streamlit

## Key Success Metrics (KPIs)
* **ROC-AUC Score:** >= 0.86 (Out-of-Time Validation)
* **Precision@Top10%:** >= 0.84
* **Business Impact:** Reduce quarterly churn from 18.6% to 15.1%
