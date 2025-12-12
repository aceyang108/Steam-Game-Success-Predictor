import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_PATH, exist_ok=True)

class SteamNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SteamNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),            
            nn.Dropout(0.3),     
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)


def load_data(use_full_data=True): #modify: Switch on different dataset 
    filename = 'training_data_full.csv' if use_full_data else 'training_data_main.csv'
    data_path = os.path.join(PROCESSED_DATA_PATH, filename)
    print(f"📂 Loading Dataset: {filename}")
    return pd.read_csv(data_path)

def train_nn():
    print("🧠 [PyTorch] Loading data...")
    df = load_data()
    
    drop_cols = ['appid', 'name', 'release_date', 'popularity', 'success_level']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df['success_level']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data standardized (Mean=0, Std=1)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights_tensor = torch.FloatTensor(class_weights)
    print(f"⚖️ Class Weights: {class_weights}")
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.values))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.values))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    model = SteamNet(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    epochs = 50
    print(f"Start Training")
    
    train_losses = []
    
    model.train() 
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()          
            outputs = model(batch_X)        
            loss = criterion(outputs, batch_y) 
            loss.backward()                
            optimizer.step()               
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Neural Network Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print("\nEvaluating Model...")
    model.eval() 
    
    all_preds = []
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        all_preds = preds.numpy()
        
    acc = accuracy_score(y_test, all_preds)
    print(f"\nNeural Network Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, all_preds))
    
    cm = confusion_matrix(y_test, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title('Neural Network Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    save_path = os.path.join(MODEL_PATH, 'pytorch_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}") #models/pytorch_model.pth

if __name__ == "__main__":
    train_nn()