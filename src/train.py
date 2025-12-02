import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_PATH, exist_ok=True)


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False



#Parameter modify
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'xgb_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'multi:softmax', 
        'num_class': 3,
        'eval_metric': 'mlogloss'
    }
}

def comprehensive_evaluation(model, X_test, y_test, X_train, y_train):
    """
    全面評估功能
    """

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cold(0)', 'Normal(1)', 'Hot(2)'],
                yticklabels=['Cold(0)', 'Normal(1)', 'Hot(2)'])
    plt.title('(Confusion Matrix)')
    plt.xlabel('prdict')
    plt.ylabel('True')
    plt.show() 
   
    print("\n(Precision/Recall):")
    print(classification_report(y_test, y_pred))

    print(" 5-Fold Cross Validation")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Score: {cv_scores}")
    print(f"Avg: {cv_scores.mean():.4f} (robust +/- {cv_scores.std() * 2:.4f})")

def load_data():
    data_path = os.path.join(PROCESSED_DATA_PATH, 'training_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError("Can't find training_data.csv")
    return pd.read_csv(data_path)

def train():
    df = load_data()
    
    if 'success_level' not in df.columns:
        raise ValueError("Can't find 'success_level' ")
        
    y = df['success_level']
    X = df.drop(columns=['success_level']) 
    if 'appid' in X.columns:
        X = X.drop(columns=['appid'])

    X = X.select_dtypes(include=['number'])

    print(f"Feature: {list(X.columns)}")
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'],
        stratify=y
    )
    

    model = xgb.XGBClassifier(**CONFIG['xgb_params'])
    model.fit(X_train, y_train)
    
    comprehensive_evaluation(model, X_test, y_test, X_train, y_train)
    
    save_file = os.path.join(MODEL_PATH, 'xgb_model.json')
    model.save_model(save_file)
    print(f"\nmodel saved: {save_file}")

if __name__ == "__main__":
    train()