import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_PATH, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'xgb_params': {
        'n_estimators': 200,        
        'max_depth': 6,
        'learning_rate': 0.05, 
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',        
        'random_state': 42
    }
}


def load_data(use_full_data=True): #modify: Switch on different dataset 
    filename = 'training_data_full.csv' if use_full_data else 'training_data_main.csv'
    data_path = os.path.join(PROCESSED_DATA_PATH, filename)
    print(f"📂 Loading Dataset: {filename}")
    return pd.read_csv(data_path)


def prepare_xy(df: pd.DataFrame):
    """
    Prepare feature matrix X and label vector y, with basic checks.

    準備特徵矩陣 X 與標籤向量 y，並做基本檢查。
    """
    if 'success_level' not in df.columns:
        raise ValueError("Label column 'success_level' is missing in data / 資料中找不到標籤欄位 'success_level'")

    # y: ensure it is integer type (0, 1, 2)
    # y：確保為整數型別（0, 1, 2）
    y = df['success_level'].astype(int)

    # X: drop label column and appid (appid is just an identifier)
    # X：移除標籤欄位與 appid（appid 僅作為識別用）
    X = df.drop(columns=['success_level'])
    if 'appid' in X.columns:
        X = X.drop(columns=['appid'])

    # Keep only numeric columns (including one-hot encoded features)
    # 僅保留數值型欄位（包含 one-hot 編碼後的欄位）
    X = X.select_dtypes(include=['number'])

    print("Feature count (特徵欄位數量):", X.shape[1])
    print("Example feature names (特徵欄位名稱示例):", list(X.columns)[:20], "...")
    print("Label distribution (y 分布):")
    print(
        y.value_counts().sort_index().rename(
            index={0: "Cold(0)", 1: "Normal(1)", 2: "Hot(2)"}
        )
    )

    # Sanity check for class labels
    # 類別標籤安全檢查，確認為預期的 0/1/2
    classes = sorted(y.unique())
    if classes != [0, 1, 2]:
        print("Warning: success_level classes are not [0, 1, 2], actual:", classes,
              "/ 警告：目前 success_level 類別並非預期的 [0, 1, 2]，實際為上述內容")

    return X, y


def comprehensive_evaluation(model, X_train, X_test, y_train, y_test):
    """
    Perform comprehensive evaluation: confusion matrix, classification report, and cross-validation.

    進行完整模型評估：包含混淆矩陣、classification report 與交叉驗證。
    """
    y_pred = model.predict(X_test)

    # Confusion matrix
    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Cold(0)', 'Normal(1)', 'Hot(2)'],
        yticklabels=['Cold(0)', 'Normal(1)', 'Hot(2)']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Precision / Recall / F1 report
    # 精確率 / 召回率 / F1 分數報告
    print("\nClassification Report (Precision / Recall / F1) / 分類報告：")
    print(classification_report(y_test, y_pred, digits=4))

    # 5-fold cross validation on training data (accuracy)
    # 使用訓練資料做 5 折交叉驗證（評估指標為 accuracy）
    print("\n5-Fold Cross Validation (accuracy) / 5 折交叉驗證（準確率）：")
    xgb_for_cv = xgb.XGBClassifier(**CONFIG['xgb_params'])
    cv_scores = cross_val_score(xgb_for_cv, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Scores: {cv_scores}")
    print(f"Avg: {cv_scores.mean():.4f}  (± {cv_scores.std() * 2:.4f})")


def train():
    """
    Main training entry point:
    1. Load data
    2. Prepare X / y
    3. Train-test split
    4. Train XGBoost model
    5. Evaluate and save model

    訓練主流程：
    1. 載入資料
    2. 準備 X / y
    3. 切分訓練 / 測試集
    4. 訓練 XGBoost 模型
    5. 評估並儲存模型
    """
    # 1. Load data
    # 1. 載入資料
    df = load_data()

    # 2. Prepare X / y
    # 2. 準備特徵與標籤
    X, y = prepare_xy(df)

    # 3. Train-test split with stratified sampling to keep class proportions
    # 3. 以 stratify 方式切分訓練 / 測試集，維持類別比例
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )

    # 4. Build and train the XGBoost classifier
    # 4. 建立並訓練 XGBoost 分類模型
    model = xgb.XGBClassifier(**CONFIG['xgb_params'])
    model.fit(X_train, y_train)

    # 5. Evaluation
    # 5. 模型評估
    comprehensive_evaluation(model, X_train, X_test, y_train, y_test)

    # 6. Save trained model to file
    # 6. 將訓練好的模型儲存成檔案
    save_file = os.path.join(MODEL_PATH, 'xgb_model.json')
    model.save_model(save_file)
    print(f"\nModel saved to: {save_file} / 模型已儲存至此路徑") #models/xgb_model.json


if __name__ == "__main__":
    train()