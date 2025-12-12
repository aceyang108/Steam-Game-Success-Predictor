import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_PATH, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'class_names': ['Cold', 'Normal', 'Hot'], 
    'xgb_params': {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softmax', 
        'num_class': 3,            
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42
    }
}

def load_data(use_full_data=True): #use_full_data => switch A/B testing 
    """
    """
    filename = 'training_data_full.csv' if use_full_data else 'training_data_main.csv'
    data_path = os.path.join(PROCESSED_DATA_PATH, filename)
    
    print(f"Loading Dataset: {filename}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f" can't find file: {data_path}。src/preprocessing.py！")
        
    return pd.read_csv(data_path)

def prepare_xy(df: pd.DataFrame):
    """prepare feature and label"""
    if 'success_level' not in df.columns:
        raise ValueError("can't find success_level'")

    y = df['success_level'].astype(int)
    drop_cols = ['success_level', 'appid', 'name', 'release_date', 'popularity']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number'])

    print(f"Features ready: {X.shape[1]} columns")
    print(f"Label Distribution:\n{y.value_counts().sort_index()}")
    unique_classes = y.unique()
    if len(unique_classes) > CONFIG['xgb_params']['num_class']:
        raise ValueError(f"ERROR class number ({len(unique_classes)}) > model ({CONFIG['xgb_params']['num_class']})！")

    return X, y

def comprehensive_evaluation(model, X_train, X_test, y_train, y_test):
    """Eval"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    class_names = CONFIG['class_names']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    print("🔄 Running 5-Fold Cross Validation...")
    xgb_cv = xgb.XGBClassifier(**CONFIG['xgb_params'])
    cv_scores = cross_val_score(xgb_cv, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"Scores: {cv_scores}")
    print(f"🏆 Average Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

def train():
    USE_FULL_DATA = True  #modify this to switch A/B Testing
    
    df = load_data(use_full_data=USE_FULL_DATA)

    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], 
        stratify=y
    )

    print("\n🚀 Training XGBoost Model...")
    model = xgb.XGBClassifier(**CONFIG['xgb_params'])
    model.fit(X_train, y_train)

    comprehensive_evaluation(model, X_train, X_test, y_train, y_test)

    save_file = os.path.join(MODEL_PATH, 'xgb_model.json')
    model.save_model(save_file)
    print(f"\nModel saved to: {save_file}")
    print("\nTip: Run 'python src/explain_shap.py' to see feature importance!")

if __name__ == "__main__":
    train()