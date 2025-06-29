# models/dnn_pytorch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

# DNN Model Architecture
class DNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_dnn(X_train, y_train, X_test, y_test, epochs=10, batch_size=1024, learning_rate=1e-3):
    """
    Trains and evaluates a Deep Neural Network (DNN) using PyTorch.
    """
    print("--- Starting PyTorch DNN Model Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_tensor = torch.from_numpy(X_train.values).float().to(device)
    y_train_tensor = torch.from_numpy(y_train.values).float().view(-1, 1).to(device)
    X_test_tensor = torch.from_numpy(X_test.values).float().to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = DNN(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        y_pred_proba = torch.sigmoid(test_outputs).cpu().numpy().flatten()
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test set ROC AUC score: {test_auc:.4f}")

    print("--- PyTorch DNN Model Training Complete ---\n")
    return model, test_auc, y_pred_proba