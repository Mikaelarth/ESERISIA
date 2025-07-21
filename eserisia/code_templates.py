"""
ESERISIA AI - Templates de Code Ultra-Avancés
===========================================
Templates complets pour tous types de projets
"""

class EserisiaCodeTemplates:
    """Templates de code ultra-sophistiqués pour tous types de projets"""
    
    @staticmethod
    def get_prisma_schema():
        return '''// Schéma Prisma ultra-optimisé généré par ESERISIA AI
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String?
  password  String
  role      Role     @default(USER)
  posts     Post[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@map("users")
}

model Post {
  id        String   @id @default(cuid())
  title     String
  content   String?
  published Boolean  @default(false)
  author    User     @relation(fields: [authorId], references: [id])
  authorId  String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@map("posts")
}

enum Role {
  USER
  ADMIN
}'''

    @staticmethod
    def get_tailwind_config():
        return '''/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'eserisia': {
          50: '#f0f9ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        }
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}'''

    @staticmethod
    def get_typescript_config():
        return {
            "compilerOptions": {
                "target": "es5",
                "lib": ["dom", "dom.iterable", "es6"],
                "allowJs": True,
                "skipLibCheck": True,
                "strict": True,
                "noEmit": True,
                "esModuleInterop": True,
                "module": "esnext",
                "moduleResolution": "bundler",
                "resolveJsonModule": True,
                "isolatedModules": True,
                "jsx": "preserve",
                "incremental": True,
                "plugins": [{"name": "next"}],
                "paths": {
                    "@/*": ["./src/*"]
                }
            },
            "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
            "exclude": ["node_modules"]
        }

    @staticmethod
    def get_fastapi_config():
        return '''"""
Configuration ultra-avancée pour FastAPI
Générée par ESERISIA AI
"""

import secrets
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "{{PROJECT_NAME}}"
    POSTGRES_PORT: str = "5432"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Security
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Email (optionnel)
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()'''

    @staticmethod
    def get_fastapi_database():
        return '''"""
Configuration base de données PostgreSQL ultra-optimisée
Générée par ESERISIA AI
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Engine PostgreSQL avec optimisations
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30,
    echo=False  # True pour debug SQL
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency pour obtenir une session DB"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()'''

    @staticmethod
    def get_ml_data_loader():
        return '''"""
Data Loader ultra-avancé pour projets ML
Générée par ESERISIA AI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class EserisiaDataset(Dataset):
    """Dataset PyTorch ultra-optimisé"""
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

class DataLoader:
    """Chargeur de données ultra-avancé avec preprocessing automatique"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Charge les données depuis différents formats"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.csv':
            data = pd.read_csv(path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(path)
        elif path.suffix.lower() == '.json':
            data = pd.read_json(path)
        elif path.suffix.lower() == '.parquet':
            data = pd.read_parquet(path)
        else:
            raise ValueError(f"Format non supporté: {path.suffix}")
        
        logger.info(f"📊 Données chargées: {data.shape}")
        
        # Séparer features et target
        target_col = self.config.get('target_column')
        if target_col and target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            return X, y
        
        return data, None
    
    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                   fit_transform: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocessing ultra-avancé avec détection automatique"""
        
        # Gestion des valeurs manquantes
        X_processed = X.copy()
        
        # Colonnes numériques
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].mean())
        
        # Colonnes catégorielles
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        X_processed[categorical_cols] = X_processed[categorical_cols].fillna('unknown')
        
        # Encoding catégoriel
        for col in categorical_cols:
            X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Normalisation
        if fit_transform:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        # Traitement du target
        y_processed = None
        if y is not None:
            if y.dtype == 'object':
                if fit_transform:
                    y_processed = self.label_encoder.fit_transform(y)
                else:
                    y_processed = self.label_encoder.transform(y)
            else:
                y_processed = y.values
        
        logger.info(f"✅ Preprocessing terminé: {X_scaled.shape}")
        return X_scaled, y_processed
    
    def create_dataloaders(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          test_size: float = 0.2, batch_size: int = 32) -> Dict[str, DataLoader]:
        """Crée les DataLoaders pour entraînement/validation"""
        from sklearn.model_selection import train_test_split
        
        if y is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_val = train_test_split(X, test_size=test_size, random_state=42)
            y_train, y_val = None, None
        
        # Datasets
        train_dataset = EserisiaDataset(X_train, y_train)
        val_dataset = EserisiaDataset(X_val, y_val)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return {
            'train': train_loader,
            'val': val_loader,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }'''

    @staticmethod
    def get_ml_neural_network():
        return '''"""
Réseau de neurones ultra-avancé avec PyTorch
Générée par ESERISIA AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import math

class EserisiaNet(nn.Module):
    """Réseau de neurones ultra-sophistiqué avec architecture adaptative"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True,
                 residual: bool = False):
        super(EserisiaNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        
        # Fonction d'activation
        self.activation_fn = self._get_activation(activation)
        
        # Architecture modulaire
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Couche linéaire
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation_fn)
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Connexions résiduelles (si dimensions compatibles)
            if residual and prev_dim == hidden_dim and i > 0:
                self.add_residual_connection(i)
            
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialisation des poids optimisée
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Sélection de la fonction d'activation"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialisation optimisée des poids"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec optimisations"""
        return self.network(x)
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Calcul de l'importance des features via gradients"""
        x.requires_grad_(True)
        output = self.forward(x)
        
        # Gradient par rapport aux inputs
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Importance = gradient * input
        importance = (grad * x).abs().mean(dim=0)
        return importance

class TransformerNet(nn.Module):
    """Architecture Transformer pour données tabulaires"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(TransformerNet, self).__init__()
        
        # Projection d'entrée
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Encodage positionnel
        self.positional_encoding = self._create_positional_encoding(d_model)
        
        # Couches Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Couche de classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )
    
    def _create_positional_encoding(self, d_model: int, max_len: int = 1000):
        """Encodage positionnel pour Transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], 1
        
        # Projection et reshape pour transformer
        x = self.input_projection(x).unsqueeze(1)  # [batch, 1, d_model]
        
        # Ajout encodage positionnel
        x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = x.squeeze(1)  # [batch, d_model]
        return self.classifier(x)'''

# Instance pour récupérer les templates
code_templates = EserisiaCodeTemplates()
