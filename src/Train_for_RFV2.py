import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_PATH, exist_ok=True)


def load_data(use_full_data=True): #modify: Switch on different dataset 
    filename = 'training_data_full.csv' if use_full_data else 'training_data_main.csv'
    data_path = os.path.join(PROCESSED_DATA_PATH, filename)
    print(f"Loading Dataset: {filename}")
    return pd.read_csv(data_path)

def train_rf():
    print("🌲 [Random Forest] Loading data...")
    df = load_data()
    drop_cols = ['appid', 'name', 'release_date', 'popularity', 'success_level']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df['success_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    print("Starting Grid Search (Auto-Tuning)...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Best Params: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nRF Acc: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 15 
    
    plt.figure(figsize=(10, 6))
    plt.title("🌲 Random Forest: Top Feature Importances")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), X.columns[indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    save_file = os.path.join(MODEL_PATH, 'rf_modelv2.pkl')
    joblib.dump(best_model, save_file)
    print(f"💾 Best Model saved to: {save_file}") #models/rf_modelv2.pkl

if __name__ == "__main__":
    train_rf()