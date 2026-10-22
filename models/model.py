import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_save_model(data_path: str, model_save_path: str):
    df = pd.read_csv(data_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    target_col = 'Claim_Status'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    model_data = {
        'model': model,
        'feature_names': list(X.columns)
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Model saved successfully to '{model_save_path}'!")

if __name__ == "__main__":
    train_and_save_model("data/accident_claims.csv", "models/insurance_claim_model.pkl")
