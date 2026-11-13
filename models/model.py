
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)


def train_and_save_model(data_path: str, model_save_path: str):

    # =====================================================
    # 1. LOAD DATASET
    # =====================================================

    print("\nLoading dataset...")

    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)
    print("\nFirst five records:")
    print(df.head())

    # =====================================================
    # 2. CHECK TARGET COLUMN
    # =====================================================

    target_col = "Claim_Status"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")

    # Remove rows where target is missing
    df = df.dropna(subset=[target_col])

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # =====================================================
    # 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
    # =====================================================

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical columns:", numeric_cols)
    print("\nCategorical columns:", categorical_cols)

    # =====================================================
    # 4. TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # =====================================================
    # 5. NUMERICAL PREPROCESSING
    # =====================================================

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    # =====================================================
    # 6. CATEGORICAL PREPROCESSING
    # =====================================================

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

    # =====================================================
    # 7. COMBINE PREPROCESSING
    # =====================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numeric_transformer,
                numeric_cols
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_cols
            )
        ]
    )

    # =====================================================
    # 8. RANDOM FOREST MODEL
    # =====================================================

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # =====================================================
    # 9. COMPLETE MACHINE LEARNING PIPELINE
    # =====================================================

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_model)
        ]
    )

    # =====================================================
    # 10. TRAIN MODEL
    # =====================================================

    print("\nTraining Random Forest model...")

    model.fit(X_train, y_train)

    print("Training completed!")

    # =====================================================
    # 11. MAKE PREDICTIONS
    # =====================================================

    y_pred = model.predict(X_test)

    # =====================================================
    # 12. MODEL EVALUATION
    # =====================================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 50)
    print("RANDOM FOREST MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # =====================================================
    # 13. CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Random Forest - Confusion Matrix")

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 14. FEATURE IMPORTANCE
    # =====================================================

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importances = model.named_steps[
        "classifier"
    ].feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 15 Important Features:")
    print(feature_importance.head(15).to_string(index=False))

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=feature_importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 15. SAVE TRAINED PIPELINE
    # =====================================================

    os.makedirs(
        os.path.dirname(model_save_path) or ".",
        exist_ok=True
    )

    model_data = {
        "model": model,
        "target_column": target_col,
        "feature_names": list(X.columns),
        "classes": list(model.classes_)
    }

    with open(model_save_path, "wb") as f:
        pickle.dump(model_data, f)

    print("\nModel saved successfully!")
    print("Location:", model_save_path)


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    train_and_save_model(
        data_path="data/accident_claims.csv",
        model_save_path="models/insurance_claim_model.pkl"
    )
