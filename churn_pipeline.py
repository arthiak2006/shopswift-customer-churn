"""
ShopSwift Retail - Customer Churn Prediction Pipeline
Author: Project Coordinator - Data Science & Analytics
Description: End-to-end preprocessing, XGBoost training, and SHAP explainability script.
"""

import hashlib
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier


def anonymize_pii(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Hashes PII columns using SHA-256 for DPDP compliance."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()
            )
    return df_clean


def build_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Creates a Scikit-Learn ColumnTransformer for automated feature scaling and encoding."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """Trains XGBoost classifier with class imbalance handling and computes SHAP values."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        eval_metric="auc",
        random_state=42
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    full_pipeline.fit(X_train, y_train)

    y_probs = full_pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)

    threshold_top10 = np.percentile(y_probs, 90)
    y_pred_top10 = (y_probs >= threshold_top10).astype(int)
    precision_top10 = precision_score(y_test, y_pred_top10, zero_division=0)

    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Precision@Top10%: {precision_top10:.4f}")

    X_test_transformed = full_pipeline.named_steps["preprocessor"].transform(X_test)
    explainer = shap.TreeExplainer(full_pipeline.named_steps["classifier"])
    shap_values = explainer.shap_values(X_test_transformed)

    joblib.dump(full_pipeline, "churn_xgb_v1.joblib")
    return full_pipeline, shap_values


if __name__ == "__main__":
    print("ShopSwift Churn Pipeline Initialized. Ready for Airflow batch execution.")
