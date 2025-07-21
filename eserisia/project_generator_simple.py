"""
ESERISIA AI - Générateur de Projets Simplifié
===========================================
Version stable pour démonstration
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
import shutil

@dataclass
class SimpleProjectTemplate:
    name: str
    description: str
    project_type: str
    technologies: List[str]
    files: Dict[str, str]  # filename -> content
    commands: List[str]

class EserisiaProjectGeneratorSimple:
    """Générateur de projets simplifié et fonctionnel"""
    
    def __init__(self):
        self.logger = logging.getLogger("ESERISIA_GENERATOR")
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, SimpleProjectTemplate]:
        """Initialise les templates de base"""
        return {
            "web": SimpleProjectTemplate(
                name="Next.js Ultra-Pro",
                description="Application web Next.js 14 avec TypeScript et Tailwind",
                project_type="web",
                technologies=["Next.js 14", "TypeScript", "Tailwind CSS", "PostgreSQL"],
                files={
                    "package.json": json.dumps({
                        "name": "{{PROJECT_NAME}}",
                        "version": "1.0.0",
                        "scripts": {
                            "dev": "next dev",
                            "build": "next build",
                            "start": "next start"
                        },
                        "dependencies": {
                            "next": "^14.0.0",
                            "react": "^18.0.0",
                            "react-dom": "^18.0.0",
                            "@types/node": "^20.0.0",
                            "@types/react": "^18.0.0",
                            "typescript": "^5.0.0",
                            "tailwindcss": "^3.4.0"
                        }
                    }, indent=2),
                    "README.md": "# {{PROJECT_NAME}}\n\nProjet généré par ESERISIA AI 🤖\n\n## Démarrage\n\n```bash\nnpm install\nnpm run dev\n```",
                    "src/app/page.tsx": '''export default function Home() {
  return (
    <main className="min-h-screen p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold mb-4 text-blue-600">
          🚀 {{PROJECT_NAME}}
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Propulsé par ESERISIA AI - L'IDE le plus puissant au monde
        </p>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">✨ Fonctionnalités</h2>
          <ul className="space-y-2">
            <li>⚡ Next.js 14 avec App Router</li>
            <li>🎨 Tailwind CSS intégré</li>
            <li>📱 Responsive design</li>
            <li>🔒 TypeScript pour la sécurité</li>
          </ul>
        </div>
      </div>
    </main>
  )
}''',
                    "tailwind.config.js": '''module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
}''',
                    "next.config.js": "/** @type {import('next').NextConfig} */\nmodule.exports = {}",
                    "tsconfig.json": json.dumps({
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
                            "paths": {"@/*": ["./src/*"]}
                        },
                        "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
                        "exclude": ["node_modules"]
                    }, indent=2)
                },
                commands=["npm install", "npm run dev"]
            ),
            
            "api": SimpleProjectTemplate(
                name="FastAPI Ultra-Performance",
                description="API REST ultra-rapide avec FastAPI et PostgreSQL",
                project_type="api", 
                technologies=["FastAPI", "PostgreSQL", "Pydantic", "SQLAlchemy"],
                files={
                    "main.py": '''from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Configuration ESERISIA AI
app = FastAPI(
    title="{{PROJECT_NAME}} API",
    description="API générée par ESERISIA AI - L'IDE le plus puissant",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

class ItemResponse(BaseModel):
    message: str
    data: Optional[Item] = None

# In-memory storage (remplacer par PostgreSQL)
items_db: List[Item] = []

@app.get("/")
async def root():
    return {
        "message": "🚀 {{PROJECT_NAME}} API",
        "powered_by": "ESERISIA AI",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/items", response_model=List[Item])
async def get_items():
    return items_db

@app.post("/items", response_model=ItemResponse)
async def create_item(item: Item):
    item.id = len(items_db) + 1
    items_db.append(item)
    return ItemResponse(message="Item créé avec succès", data=item)

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return ItemResponse(message="Item trouvé", data=item)
    raise HTTPException(status_code=404, detail="Item non trouvé")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
                    "requirements.txt": '''fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6
sqlalchemy>=2.0.23
psycopg2-binary>=2.9.9
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
''',
                    "README.md": "# {{PROJECT_NAME}} API\n\nAPI générée par ESERISIA AI 🤖\n\n## Démarrage\n\n```bash\npip install -r requirements.txt\npython main.py\n```\n\nAPI disponible sur: http://localhost:8000\nDocumentation: http://localhost:8000/docs",
                    "Dockerfile": '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
                },
                commands=["pip install -r requirements.txt", "python main.py"]
            ),
            
            "ml": SimpleProjectTemplate(
                name="ML Project Complete",
                description="Projet Machine Learning complet avec PyTorch",
                project_type="ml",
                technologies=["PyTorch", "Pandas", "Scikit-learn", "Jupyter"],
                files={
                    "main.py": '''#!/usr/bin/env python3
"""
{{PROJECT_NAME}} - Projet ML ESERISIA AI
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
logger = logging.getLogger("{{PROJECT_NAME}}")

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
''',
                    "requirements.txt": '''torch>=2.1.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
notebook>=7.0.0
''',
                    "visualize.py": '''import matplotlib.pyplot as plt
import pandas as pd
import torch
from main import EserisiaModel

def visualize_results():
    """Visualise les résultats du modèle"""
    print("📊 Visualisation des résultats ESERISIA AI")
    
    # Charger les données
    try:
        data = pd.read_csv('data.csv')
        
        # Graphique des données
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(data['target'], bins=30, alpha=0.7, color='blue')
        plt.title('Distribution de la variable cible')
        plt.xlabel('Valeur')
        plt.ylabel('Fréquence')
        
        plt.subplot(1, 2, 2)
        correlation_matrix = data.corr()
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('Matrice de corrélation')
        plt.xticks(range(len(data.columns)), data.columns, rotation=45)
        plt.yticks(range(len(data.columns)), data.columns)
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Graphiques sauvegardés: data_analysis.png")
        
    except Exception as e:
        print(f"❌ Erreur lors de la visualisation: {e}")

if __name__ == "__main__":
    visualize_results()
''',
                    "README.md": '''# {{PROJECT_NAME}} - Projet ML ESERISIA AI

Projet Machine Learning ultra-avancé généré automatiquement par ESERISIA AI 🤖

## 🚀 Démarrage rapide

### Installation
```bash
pip install -r requirements.txt
```

### Entraînement
```bash
python main.py
```

### Visualisation
```bash
python visualize.py
```

## 📊 Structure

- `main.py`: Script principal d'entraînement
- `visualize.py`: Script de visualisation des résultats
- `model.pth`: Modèle entraîné (généré automatiquement)
- `data.csv`: Données d'entraînement (générées si inexistantes)

## 🧠 Modèle

Réseau de neurones optimisé avec:
- Architecture adaptive
- Dropout pour la régularisation  
- Optimiseur Adam
- Validation automatique

## 📈 Résultats

Le modèle génère automatiquement:
- Métriques de performance
- Graphiques de visualisation
- Sauvegarde automatique

---
Généré par **ESERISIA AI** - L'IDE le plus puissant au monde 🌟
'''
                },
                commands=["pip install -r requirements.txt", "python main.py"]
            )
        }
    
    async def generate_project(self, project_type: str, project_name: str, 
                             destination_path: str = ".", template_name: Optional[str] = None,
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Génère un projet complet"""
        
        # Sélection du template
        if project_type not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Type de projet non supporté: {project_type}. Disponibles: {available}")
        
        template = self.templates[project_type]
        
        # Création du dossier projet
        project_path = Path(destination_path) / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🏗️ Création projet {project_name} dans {project_path}")
        
        # Génération des fichiers
        files_created = 0
        project_structure = {}
        
        for file_path, content in template.files.items():
            # Remplacer les variables
            processed_content = content.replace("{{PROJECT_NAME}}", project_name)
            
            # Créer la structure de fichiers
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Écrire le fichier
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            files_created += 1
            
            # Ajouter à la structure
            parts = Path(file_path).parts
            current = project_structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = "📄"
        
        self.logger.info(f"✅ {files_created} fichiers créés")
        
        return {
            "project_name": project_name,
            "project_type": project_type,
            "project_path": str(project_path),
            "template_used": template.name,
            "files_created": files_created,
            "setup_commands": template.commands,
            "project_structure": project_structure,
            "technologies": template.technologies
        }
    
    def list_available_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Liste tous les templates disponibles"""
        templates_info = {}
        
        for key, template in self.templates.items():
            templates_info[key] = [{
                "name": template.name,
                "description": template.description,
                "technologies": template.technologies
            }]
        
        return templates_info

# Instance globale pour rétrocompatibilité
eserisia_generator = EserisiaProjectGeneratorSimple()
