
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")


def train_and_save_model(
    data_path: str,
    model_save_path: str,
    random_state: int = 42
):

    # ==================================================
    # 1. LOAD DATASET
    # ==================================================

    print("\nLoading insurance claims dataset...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)

    target_col = "Claim_Status"

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    # Remove rows with missing target values.
    df = df.dropna(subset=[target_col]).copy()

    if len(df) < 50:
        raise ValueError(
            "Dataset is too small. At least 50 records are required."
        )

    # Validate target values.
    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="raise"
    )

    if not df[target_col].isin([0, 1]).all():
        raise ValueError(
            "Claim_Status must contain only 0 and 1."
        )

    df[target_col] = df[target_col].astype(int)

    if df[target_col].nunique() != 2:
        raise ValueError(
            "Both claim classes (0 and 1) must be present."
        )

    # Remove completely empty feature columns.
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No usable input features found.")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    print("\nTarget percentages:")
    print(
        (y.value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

    # ==================================================
    # 2. IDENTIFY FEATURE TYPES
    # ==================================================

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    unsupported_cols = [
        col for col in X.columns
        if col not in numeric_cols + categorical_cols
    ]

    if unsupported_cols:
        raise ValueError(
            f"Unsupported feature columns: {unsupported_cols}"
        )

    if not numeric_cols and not categorical_cols:
        raise ValueError("No supported features found.")

    print("\nNumeric features:", numeric_cols)
    print("Categorical features:", categorical_cols)

    # ==================================================
    # 3. TRAIN-TEST SPLIT
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # ==================================================
    # 4. DATA PREPROCESSING
    # ==================================================

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_transformer, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_transformer, categorical_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # ==================================================
    # 5. RANDOM FOREST PIPELINE
    # ==================================================

    rf_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_model)
        ]
    )

    # ==================================================
    # 6. HYPERPARAMETER TUNING
    # ==================================================

    param_grid = {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 8, 12, 16, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", 0.8],
        "classifier__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_grid,
        n_iter=12,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    # ==================================================
    # 7. TRAIN MODEL
    # ==================================================

    print("\nStarting Random Forest hyperparameter tuning...")

    search.fit(X_train, y_train)

    model = search.best_estimator_

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        "\nBest Cross-Validation Macro F1:",
        round(search.best_score_, 4)
    )

    # ==================================================
    # 8. MODEL PREDICTIONS
    # ==================================================

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)

    classes = list(model.classes_)

    # Find probability column corresponding to class 1.
    positive_class_index = classes.index(1)

    y_positive_probability = y_probability[
        :, positive_class_index
    ]

    # ==================================================
    # 9. MODEL EVALUATION
    # ==================================================

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_positive_probability
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"ROC-AUC           : {auc:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Rejected (0)",
                "Approved (1)"
            ],
            zero_division=0
        )
    )

    # ==================================================
    # 10. CONFUSION MATRIX
    # ==================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")

    print(pd.DataFrame(
        cm,
        index=["Actual Rejected", "Actual Approved"],
        columns=["Predicted Rejected", "Predicted Approved"]
    ))

    tn, fp, fn, tp = cm.ravel()

    print("\nPrediction Analysis:")
    print("True Negatives :", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives :", tp)

    # ==================================================
    # 11. FEATURE IMPORTANCE
    # ==================================================

    classifier = model.named_steps["classifier"]

    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop Feature Importances:")

    print(importance_df.head(15).to_string(index=False))

    # ==================================================
    # 12. SAVE TRAINED MODEL
    # ==================================================

    os.makedirs(
        os.path.dirname(model_save_path) or ".",
        exist_ok=True
    )

    model_data = {
        "model": model,
        "target_column": target_col,
        "feature_names": list(X.columns),
        "classes": classes,
        "best_parameters": search.best_params_,
        "cross_validation_f1_macro": search.best_score_,
        "evaluation_metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc
        },
        "random_state": random_state
    }

    with open(model_save_path, "wb") as f:
        pickle.dump(model_data, f)

    print("\n" + "=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)

    print("Model path:", model_save_path)

    return model_data


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":

    train_and_save_model(
        data_path="data/accident_claims.csv",
        model_save_path="models/insurance_claim_model.pkl",
        random_state=42
    )
