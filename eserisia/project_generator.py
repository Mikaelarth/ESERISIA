"""
ESERISIA AI - Générateur de Projets Ultra-Avancé
===============================================
Capable de créer tout type de projet avec architecture optimale
Templates : Web, Mobile, Desktop, IA, Blockchain, GameDev, etc.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import json
import yaml
import subprocess
from datetime import datetime
import shutil
from jinja2 import Template, Environment, FileSystemLoader

try:
    from .code_templates import code_templates
except ImportError:
    code_templates = None

logger = logging.getLogger(__name__)

@dataclass
class ProjectTemplate:
    """Template de projet ultra-avancé"""
    name: str
    category: str
    description: str
    technologies: List[str]
    structure: Dict[str, Any]
    dependencies: Dict[str, List[str]]
    setup_commands: List[str]
    templates: Dict[str, str]
    complexity: str = "Medium"
    estimated_time: str = "30 minutes"

class EserisiaProjectGenerator:
    """
    Générateur de projets ultra-avancé ESERISIA AI
    Capable de créer n'importe quel type de projet
    """
    
    def __init__(self):
        self.templates_path = Path(__file__).parent / "templates"
        self.templates_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Templates disponibles
        self.project_templates = {
            # 🌐 PROJETS WEB
            "nextjs-advanced": ProjectTemplate(
                name="Next.js Ultra-Avancé",
                category="web",
                description="Application Next.js 14+ avec TypeScript, Tailwind, API Routes",
                technologies=["Next.js", "TypeScript", "Tailwind CSS", "Prisma", "NextAuth"],
                structure=self._get_nextjs_structure(),
                dependencies={"npm": ["next", "react", "typescript", "@types/react", "tailwindcss", "prisma"]},
                setup_commands=["npm install", "npx prisma generate", "npm run dev"],
                templates={}
            ),
            
            "fastapi-ultra": ProjectTemplate(
                name="FastAPI Ultra-Performance",
                category="backend",
                description="API ultra-rapide avec FastAPI, PostgreSQL, Redis, JWT",
                technologies=["FastAPI", "PostgreSQL", "Redis", "SQLAlchemy", "JWT", "Docker"],
                structure=self._get_fastapi_structure(),
                dependencies={"pip": ["fastapi", "uvicorn", "sqlalchemy", "psycopg2", "redis", "python-jose"]},
                setup_commands=["pip install -r requirements.txt", "alembic upgrade head", "uvicorn main:app --reload"],
                templates={}
            ),
            
            # 🤖 PROJETS IA/ML
            "ai-ml-complete": ProjectTemplate(
                name="Projet IA/ML Complet",
                category="ai",
                description="Pipeline ML complet avec MLOps, monitoring, déploiement",
                technologies=["PyTorch", "TensorFlow", "MLflow", "Docker", "Kubernetes", "FastAPI"],
                structure=self._get_ai_ml_structure(),
                dependencies={"pip": ["torch", "tensorflow", "mlflow", "fastapi", "pandas", "numpy", "scikit-learn"]},
                setup_commands=["pip install -r requirements.txt", "mlflow server --host 0.0.0.0", "python train.py"],
                templates={}
            ),
            
            # 📱 PROJETS MOBILE
            "react-native-pro": ProjectTemplate(
                name="React Native Pro",
                category="mobile",
                description="App mobile cross-platform avec navigation, état global",
                technologies=["React Native", "TypeScript", "Redux Toolkit", "React Navigation", "Expo"],
                structure=self._get_react_native_structure(),
                dependencies={"npm": ["react-native", "typescript", "@reduxjs/toolkit", "react-navigation"]},
                setup_commands=["npm install", "npx pod-install ios", "npm run start"],
                templates={}
            ),
            
            # 🎮 PROJETS GAMEDEV
            "unity-3d-advanced": ProjectTemplate(
                name="Unity 3D Avancé",
                category="gamedev",
                description="Jeu 3D Unity avec systèmes avancés, networking",
                technologies=["Unity", "C#", "Mirror Networking", "Addressables", "Timeline"],
                structure=self._get_unity_structure(),
                dependencies={},
                setup_commands=["Unity Hub", "Open Project", "Package Manager"],
                templates={}
            ),
            
            # 🔗 BLOCKCHAIN
            "blockchain-dapp": ProjectTemplate(
                name="DApp Blockchain Complète",
                category="blockchain",
                description="Application décentralisée avec smart contracts, front-end",
                technologies=["Solidity", "Hardhat", "React", "Ethers.js", "IPFS", "MetaMask"],
                structure=self._get_blockchain_structure(),
                dependencies={"npm": ["hardhat", "ethers", "react", "@openzeppelin/contracts"]},
                setup_commands=["npm install", "npx hardhat compile", "npx hardhat deploy"],
                templates={}
            ),
            
            # 🖥️ DESKTOP
            "electron-advanced": ProjectTemplate(
                name="Electron Advanced Desktop",
                category="desktop",
                description="Application desktop multi-plateforme avec auto-updater",
                technologies=["Electron", "TypeScript", "React", "Electron Builder", "Auto Updater"],
                structure=self._get_electron_structure(),
                dependencies={"npm": ["electron", "react", "typescript", "electron-builder"]},
                setup_commands=["npm install", "npm run dev", "npm run build"],
                templates={}
            ),
            
            # 🛠️ DevOps/Infrastructure
            "devops-complete": ProjectTemplate(
                name="DevOps Infrastructure Complète",
                category="devops",
                description="Infrastructure as Code avec monitoring complet",
                technologies=["Docker", "Kubernetes", "Terraform", "Prometheus", "Grafana", "Jenkins"],
                structure=self._get_devops_structure(),
                dependencies={},
                setup_commands=["terraform init", "docker-compose up", "kubectl apply -f k8s/"],
                templates={}
            )
        }
        
        self.logger.info("🚀 Générateur de projets ESERISIA initialisé")
    
    async def create_project(self, 
                           project_name: str,
                           template_key: str,
                           output_path: str = ".",
                           custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Créer un projet complet à partir d'un template
        """
        self.logger.info(f"🏗️ Création projet: {project_name} ({template_key})")
        
        if template_key not in self.project_templates:
            return {
                "success": False,
                "error": f"Template '{template_key}' non trouvé",
                "available": list(self.project_templates.keys())
            }
        
        template = self.project_templates[template_key]
        project_path = Path(output_path) / project_name
        
        try:
            # Créer structure du projet
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Générer structure de fichiers
            await self._generate_project_structure(project_path, template, project_name, custom_config)
            
            # Créer fichiers de configuration
            await self._generate_config_files(project_path, template, project_name)
            
            # Générer code source
            await self._generate_source_code(project_path, template, project_name, custom_config)
            
            # Créer documentation
            await self._generate_documentation(project_path, template, project_name)
            
            # Setup développement
            if custom_config and custom_config.get("auto_setup", True):
                await self._setup_development_environment(project_path, template)
            
            result = {
                "success": True,
                "project_name": project_name,
                "project_path": str(project_path),
                "template": template_key,
                "technologies": template.technologies,
                "next_steps": template.setup_commands,
                "estimated_setup_time": template.estimated_time,
                "files_created": await self._count_files_created(project_path)
            }
            
            self.logger.info(f"✅ Projet '{project_name}' créé avec succès!")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erreur création projet: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_path": str(project_path)
            }
    
    async def _generate_project_structure(self, project_path: Path, template: ProjectTemplate, 
                                        project_name: str, custom_config: Optional[Dict[str, Any]]):
        """Génère la structure de dossiers du projet"""
        structure = template.structure
        
        def create_structure(base_path: Path, struct: Dict[str, Any]):
            for name, content in struct.items():
                path = base_path / name
                
                if isinstance(content, dict):
                    path.mkdir(exist_ok=True)
                    create_structure(path, content)
                elif isinstance(content, str):
                    # Fichier avec contenu template
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        # Remplacer variables template
                        content_rendered = content.replace("{{PROJECT_NAME}}", project_name)
                        content_rendered = content_rendered.replace("{{DATE}}", datetime.now().isoformat())
                        f.write(content_rendered)
                else:
                    # Dossier vide
                    path.mkdir(exist_ok=True)
        
        create_structure(project_path, structure)
    
    async def _generate_config_files(self, project_path: Path, template: ProjectTemplate, project_name: str):
        """Génère les fichiers de configuration"""
        
        # package.json pour projets Node.js
        if any(tech in template.technologies for tech in ["Next.js", "React", "React Native", "Electron"]):
            package_json = {
                "name": project_name.lower().replace(" ", "-"),
                "version": "1.0.0",
                "description": template.description,
                "main": "index.js" if "Electron" in template.technologies else "next.config.js",
                "scripts": self._get_npm_scripts(template),
                "dependencies": {},
                "devDependencies": {},
                "keywords": template.technologies,
                "author": "ESERISIA AI Generator",
                "license": "MIT"
            }
            
            with open(project_path / "package.json", 'w') as f:
                json.dump(package_json, f, indent=2)
        
        # requirements.txt pour projets Python
        if any(tech in template.technologies for tech in ["FastAPI", "PyTorch", "TensorFlow"]):
            requirements = template.dependencies.get("pip", [])
            with open(project_path / "requirements.txt", 'w') as f:
                f.write('\n'.join(requirements))
        
        # Docker files
        await self._generate_docker_files(project_path, template, project_name)
        
        # CI/CD files
        await self._generate_cicd_files(project_path, template)
    
    def _get_npm_scripts(self, template: ProjectTemplate) -> Dict[str, str]:
        """Génère les scripts npm selon le template"""
        base_scripts = {
            "dev": "next dev" if "Next.js" in template.technologies else "npm start",
            "build": "next build" if "Next.js" in template.technologies else "npm run build",
            "start": "next start" if "Next.js" in template.technologies else "node index.js",
            "lint": "eslint . --ext .ts,.tsx,.js,.jsx",
            "test": "jest",
            "type-check": "tsc --noEmit"
        }
        
        if "React Native" in template.technologies:
            base_scripts.update({
                "android": "react-native run-android",
                "ios": "react-native run-ios",
                "start": "react-native start"
            })
        
        if "Electron" in template.technologies:
            base_scripts.update({
                "electron": "electron .",
                "electron-dev": "NODE_ENV=development electron .",
                "dist": "electron-builder"
            })
        
        return base_scripts
    
    # Templates de structure pour chaque type de projet
    def _get_nextjs_structure(self) -> Dict[str, Any]:
        return {
            "src": {
                "app": {
                    "globals.css": "/* Styles globaux générés par ESERISIA AI */\n@tailwind base;\n@tailwind components;\n@tailwind utilities;",
                    "layout.tsx": self._get_nextjs_layout_template(),
                    "page.tsx": self._get_nextjs_page_template(),
                    "api": {
                        "auth": {},
                        "users": {}
                    }
                },
                "components": {
                    "ui": {},
                    "forms": {},
                    "layout": {}
                },
                "lib": {
                    "db.ts": "// Configuration base de données",
                    "auth.ts": "// Configuration authentification",
                    "utils.ts": "// Utilitaires"
                },
                "types": {},
                "hooks": {}
            },
            "public": {},
            "prisma": {
                "schema.prisma": self._get_prisma_schema() if code_templates else "// Prisma schema placeholder"
            },
            "tailwind.config.js": self._get_tailwind_config() if hasattr(self, '_get_tailwind_config') else "// Tailwind config",
            "next.config.js": "/** @type {import('next').NextConfig} */\nmodule.exports = {}",
            "tsconfig.json": json.dumps(self._get_typescript_config() if hasattr(self, '_get_typescript_config') else {}, indent=2),
            ".env.example": "DATABASE_URL=postgresql://user:password@localhost:5432/{{PROJECT_NAME}}\nNEXTAUTH_SECRET=your-secret"
        }
    
    def _get_fastapi_structure(self) -> Dict[str, Any]:
        return {
            "app": {
                "main.py": self._get_fastapi_main_template(),
                "api": {
                    "v1": {
                        "__init__.py": "",
                        "endpoints": {
                            "__init__.py": "",
                            "users.py": self._get_fastapi_users_endpoint(),
                            "auth.py": self._get_fastapi_auth_endpoint()
                        }
                    }
                },
                "core": {
                    "__init__.py": "",
                    "config.py": self._get_fastapi_config(),
                    "security.py": self._get_fastapi_security(),
                    "database.py": self._get_fastapi_database()
                },
                "models": {
                    "__init__.py": "",
                    "user.py": self._get_fastapi_user_model()
                },
                "schemas": {
                    "__init__.py": "",
                    "user.py": self._get_fastapi_user_schema()
                },
                "services": {
                    "__init__.py": "",
                    "user_service.py": ""
                }
            },
            "alembic": {},
            "tests": {
                "__init__.py": "",
                "test_main.py": self._get_fastapi_test_template()
            },
            "Dockerfile": self._get_fastapi_dockerfile(),
            "docker-compose.yml": self._get_fastapi_docker_compose(),
            ".env.example": "DATABASE_URL=postgresql://user:password@localhost/{{PROJECT_NAME}}\nSECRET_KEY=your-secret-key\nREDIS_URL=redis://localhost:6379"
        }
    
    def _get_ai_ml_structure(self) -> Dict[str, Any]:
        return {
            "src": {
                "data": {
                    "__init__.py": "",
                    "data_loader.py": self._get_ml_data_loader(),
                    "preprocessing.py": self._get_ml_preprocessing()
                },
                "models": {
                    "__init__.py": "",
                    "base_model.py": self._get_ml_base_model(),
                    "neural_network.py": self._get_ml_neural_network()
                },
                "training": {
                    "__init__.py": "",
                    "trainer.py": self._get_ml_trainer(),
                    "evaluator.py": self._get_ml_evaluator()
                },
                "inference": {
                    "__init__.py": "",
                    "predictor.py": self._get_ml_predictor()
                },
                "utils": {
                    "__init__.py": "",
                    "metrics.py": self._get_ml_metrics(),
                    "visualization.py": self._get_ml_visualization()
                }
            },
            "notebooks": {
                "exploration.ipynb": "// Notebook d'exploration des données",
                "training.ipynb": "// Notebook d'entraînement",
                "evaluation.ipynb": "// Notebook d'évaluation"
            },
            "config": {
                "model_config.yaml": self._get_ml_model_config(),
                "training_config.yaml": self._get_ml_training_config()
            },
            "api": {
                "main.py": self._get_ml_api_main(),
                "models.py": self._get_ml_api_models()
            },
            "docker": {
                "Dockerfile.train": self._get_ml_dockerfile_train(),
                "Dockerfile.serve": self._get_ml_dockerfile_serve()
            },
            "mlflow": {},
            "models": {},  # Modèles sauvegardés
            "data": {
                "raw": {},
                "processed": {},
                "external": {}
            }
        }
    
    # Templates de code ultra-avancés
    def _get_nextjs_layout_template(self) -> str:
        return '''import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '{{PROJECT_NAME}} - Powered by ESERISIA AI',
  description: 'Application ultra-avancée générée par ESERISIA AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          <nav className="bg-white shadow-sm border-b">
            <div className="container mx-auto px-4 py-4">
              <h1 className="text-xl font-bold text-gray-800">{{PROJECT_NAME}}</h1>
            </div>
          </nav>
          <main className="container mx-auto px-4 py-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}'''
    
    def _get_nextjs_page_template(self) -> str:
        return '''export default function Home() {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          🚀 {{PROJECT_NAME}}
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Application ultra-avancée générée par ESERISIA AI
        </p>
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-8 rounded-lg">
          <h2 className="text-2xl font-bold mb-4">✨ Fonctionnalités</h2>
          <ul className="space-y-2 text-left">
            <li>✅ Next.js 14+ avec App Router</li>
            <li>✅ TypeScript pour la sécurité de type</li>
            <li>✅ Tailwind CSS pour le styling</li>
            <li>✅ Prisma pour la base de données</li>
            <li>✅ NextAuth pour l'authentification</li>
            <li>✅ Architecture ultra-optimisée</li>
          </ul>
        </div>
      </div>
    </div>
  )
}'''

    def _get_fastapi_main_template(self) -> str:
        return '''"""
{{PROJECT_NAME}} - API Ultra-Avancée
Générée par ESERISIA AI
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.api.v1.endpoints import users, auth

# Application FastAPI ultra-optimisée
app = FastAPI(
    title="{{PROJECT_NAME}} API",
    description="API ultra-avancée générée par ESERISIA AI",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Middleware de sécurité
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes API
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])

@app.get("/")
async def root():
    return {
        "message": "🚀 {{PROJECT_NAME}} API - Powered by ESERISIA AI",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Ultra-fast FastAPI",
            "PostgreSQL database",
            "Redis caching",
            "JWT authentication",
            "Docker ready",
            "Production optimized"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "{{PROJECT_NAME}}"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )'''

    # Continuer avec plus de templates...
    async def get_available_templates(self) -> Dict[str, Any]:
        """Retourne tous les templates disponibles"""
        return {
            "categories": {
                "web": [k for k, v in self.project_templates.items() if v.category == "web"],
                "backend": [k for k, v in self.project_templates.items() if v.category == "backend"],
                "ai": [k for k, v in self.project_templates.items() if v.category == "ai"],
                "mobile": [k for k, v in self.project_templates.items() if v.category == "mobile"],
                "gamedev": [k for k, v in self.project_templates.items() if v.category == "gamedev"],
                "blockchain": [k for k, v in self.project_templates.items() if v.category == "blockchain"],
                "desktop": [k for k, v in self.project_templates.items() if v.category == "desktop"],
                "devops": [k for k, v in self.project_templates.items() if v.category == "devops"]
            },
            "templates": {
                k: {
                    "name": v.name,
                    "description": v.description,
                    "technologies": v.technologies,
                    "complexity": v.complexity,
                    "estimated_time": v.estimated_time
                }
                for k, v in self.project_templates.items()
            },
            "total_templates": len(self.project_templates)
        }

# Instance globale
eserisia_generator = EserisiaProjectGenerator()

# Fonctions utilitaires
async def create_project_ultra_advanced(project_name: str, template_key: str, 
                                      output_path: str = ".", config: Dict[str, Any] = None):
    """Créer un projet ultra-avancé avec ESERISIA AI"""
    return await eserisia_generator.create_project(project_name, template_key, output_path, config)

async def list_available_templates():
    """Lister tous les templates disponibles"""
    return await eserisia_generator.get_available_templates()

def get_project_recommendations(requirements: Dict[str, Any]) -> List[str]:
    """Recommandations de templates basées sur les besoins"""
    recommendations = []
    
    if requirements.get("type") == "web":
        if requirements.get("complexity") == "high":
            recommendations.append("nextjs-advanced")
        else:
            recommendations.append("nextjs-advanced")  # Toujours le meilleur pour le web
    
    if requirements.get("type") == "api":
        recommendations.append("fastapi-ultra")
    
    if requirements.get("type") == "ai" or "machine learning" in requirements.get("keywords", []):
        recommendations.append("ai-ml-complete")
    
    return recommendations or ["nextjs-advanced"]  # Fallback

__all__ = [
    'EserisiaProjectGenerator', 'ProjectTemplate', 'eserisia_generator',
    'create_project_ultra_advanced', 'list_available_templates', 
    'get_project_recommendations'
]
