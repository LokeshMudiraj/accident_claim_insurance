import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_charts(csv_path: str):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Claim_Status', data=df, palette='Blues_d')
    plt.title("Claim Approval vs Denial Distribution")
    plt.xlabel("Claim Status (0 = Denied, 1 = Approved)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("claim_status_distribution.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()
    
    print("EDA Visualizations saved successfully!")

if __name__ == "__main__":
    generate_eda_charts("../data/accident_claims.csv")
