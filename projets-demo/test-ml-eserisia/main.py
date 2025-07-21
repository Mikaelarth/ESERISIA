#!/usr/bin/env python3
"""
test-ml-eserisia - Projet ML ESERISIA AI
=======================================
Projet ML ultra-avancé généré automatiquement
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-ml-eserisia")

class EserisiaModel(nn.Module):
    """Modèle neural network optimisé par ESERISIA AI"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 1):
        super(EserisiaModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def load_data(file_path: str = "data.csv"):
    """Charge les données"""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"📊 Données chargées: {data.shape}")
        return data
    except FileNotFoundError:
        logger.warning("⚠️ Fichier data.csv non trouvé. Création de données factices...")
        # Données d'exemple
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 4)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 - X[:, 2] + np.random.randn(n_samples) * 0.1
        
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        data['target'] = y
        data.to_csv('data.csv', index=False)
        
        return data

def train_model(X_train, y_train, X_val, y_val):
    """Entraîne le modèle"""
    model = EserisiaModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Entraînement
    train_losses = []
    val_losses = []
    
    for epoch in range(100):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/100], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    return model, train_losses, val_losses

def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage du projet ML ESERISIA AI")
    
    # Chargement des données
    data = load_data()
    
    # Préparation des données
    feature_columns = [col for col in data.columns if col != 'target']
    X = data[feature_columns].values
    y = data['target'].values
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    logger.info(f"📊 Tailles des ensembles - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Entraînement
    model, train_losses, val_losses = train_model(X_train, y_train, X_val, y_val)
    
    # Évaluation finale
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        test_predictions = model(X_test_tensor)
        test_mse = nn.MSELoss()(test_predictions, torch.FloatTensor(y_test).unsqueeze(1))
        logger.info(f"🎯 MSE Test Final: {test_mse.item():.4f}")
    
    # Sauvegarde du modèle
    torch.save(model.state_dict(), 'model.pth')
    logger.info("💾 Modèle sauvegardé: model.pth")
    
    print("✅ Entraînement terminé avec succès!")
    print("📈 Utilisez 'python visualize.py' pour voir les résultats")

if __name__ == "__main__":
    main()
