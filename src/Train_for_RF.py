import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models')


def load_data(use_full_data=True): #modify: Switch on different dataset 
    filename = 'training_data_full.csv' if use_full_data else 'training_data_main.csv'
    data_path = os.path.join(PROCESSED_DATA_PATH, filename)
    print(f"📂 Loading Dataset: {filename}")
    return pd.read_csv(data_path)

def train_rf():
    print("[Random Forest] Loading data...")
    df = load_data()
    drop_cols = ['appid', 'name', 'release_date', 'popularity', 'success_level']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df['success_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Random Forest Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    save_file = os.path.join(MODEL_PATH, 'rf_model.pkl')
    joblib.dump(rf_model, save_file)
    print(f"💾 Model saved to: {save_file}") #models/rf_model.pkl

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Random Forest Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    train_rf()