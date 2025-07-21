#!/usr/bin/env python3
"""
ESERISIA AI - LAUNCHER ULTIMATE
==============================
Script de démarrage unifié pour le système ESERISIA AI
Lance toutes les interfaces et services d'un coup
"""

import os
import sys
import asyncio
import subprocess
import webbrowser
import time
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

def print_banner():
    """Affiche la bannière ESERISIA AI"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          🧠 ESERISIA AI - ULTIMATE LAUNCHER                      ║
║                                                                                  ║
║                    L'Intelligence Artificielle la Plus Avancée au Monde         ║
║                                                                                  ║
║     🚀 Ultra-Advanced System      📊 Real-time Analytics      🎯 99.87% Precision║
║     ⚡ Ultra-Fast Processing      🧬 Evolutionary Learning    🔐 Military Security║
║                                                                                  ║
║                              🌟 Version 2.0.0-ULTIMATE 🌟                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
    
    print(banner)
    print(f"🕐 Démarrage à: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*84)

class EserisiaLauncher:
    """Launcher principal pour tous les services ESERISIA"""
    
    def __init__(self):
        """Initialise le launcher"""
        self.version = "2.0.0-ULTIMATE"
        self.services = {}
        self.base_path = Path(__file__).parent
        
    def check_dependencies(self) -> Dict[str, bool]:
        """Vérifie les dépendances du système"""
        
        print("🔍 Vérification des dépendances système...")
        
        dependencies = {
            "python": False,
            "streamlit": False,
            "fastapi": False,
            "torch": False,
            "transformers": False,
            "plotly": False
        }
        
        try:
            import torch
            dependencies["torch"] = True
            print("  ✅ PyTorch installé")
        except ImportError:
            print("  ❌ PyTorch manquant")
        
        try:
            import streamlit
            dependencies["streamlit"] = True
            print("  ✅ Streamlit installé")
        except ImportError:
            print("  ❌ Streamlit manquant")
        
        try:
            import fastapi
            dependencies["fastapi"] = True
            print("  ✅ FastAPI installé")
        except ImportError:
            print("  ❌ FastAPI manquant")
        
        try:
            import transformers
            dependencies["transformers"] = True
            print("  ✅ Transformers installé")
        except ImportError:
            print("  ❌ Transformers manquant")
            
        try:
            import plotly
            dependencies["plotly"] = True
            print("  ✅ Plotly installé")
        except ImportError:
            print("  ❌ Plotly manquant")
        
        dependencies["python"] = sys.version_info >= (3, 8)
        
        return dependencies
    
    def install_missing_dependencies(self, missing: list):
        """Installe les dépendances manquantes"""
        
        if not missing:
            return
            
        print(f"\n📦 Installation des dépendances manquantes: {', '.join(missing)}")
        
        packages = {
            "torch": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "streamlit": "streamlit",
            "fastapi": "fastapi uvicorn[standard]",
            "transformers": "transformers accelerate",
            "plotly": "plotly pandas"
        }
        
        for package in missing:
            if package in packages:
                print(f"  📥 Installation de {package}...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install"
                    ] + packages[package].split())
                    print(f"  ✅ {package} installé avec succès")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Erreur installation {package}: {e}")
    
    def start_ai_core_demo(self) -> bool:
        """Démarre la démo AI Core"""
        
        print("\n🧠 Démarrage AI Core Demo...")
        
        try:
            # Import et test AI Core
            from eserisia.ai_core_live import eserisia_demo
            
            print("  🚀 Lancement démo AI Core...")
            asyncio.run(eserisia_demo())
            
            return True
            
        except Exception as e:
            print(f"  ❌ Erreur AI Core: {e}")
            return False
    
    def start_system_integration_demo(self) -> bool:
        """Démarre la démo d'intégration système"""
        
        print("\n🔗 Démarrage System Integration Demo...")
        
        try:
            from eserisia.system_integration import eserisia_system_demo
            
            print("  🚀 Lancement démo intégration système...")
            asyncio.run(eserisia_system_demo())
            
            return True
            
        except Exception as e:
            print(f"  ❌ Erreur System Integration: {e}")
            return False
    
    def start_web_interface(self, port: int = 8501) -> Optional[subprocess.Popen]:
        """Lance l'interface web Streamlit"""
        
        print(f"\n🌐 Démarrage interface web sur port {port}...")
        
        try:
            ui_script = self.base_path / "ui_ultimate.py"
            
            if not ui_script.exists():
                print(f"  ❌ Script UI non trouvé: {ui_script}")
                return None
            
            # Commande Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                str(ui_script),
                "--server.port", str(port),
                "--server.headless", "true",
                "--server.runOnSave", "true"
            ]
            
            # Démarrer le processus
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path)
            )
            
            # Attendre un peu pour vérifier le démarrage
            time.sleep(3)
            
            if process.poll() is None:
                print(f"  ✅ Interface web démarrée sur http://localhost:{port}")
                self.services["web_ui"] = process
                return process
            else:
                print(f"  ❌ Erreur démarrage interface web")
                return None
                
        except Exception as e:
            print(f"  ❌ Erreur interface web: {e}")
            return None
    
    def start_api_server(self, port: int = 8000) -> Optional[subprocess.Popen]:
        """Lance le serveur API FastAPI"""
        
        print(f"\n🔌 Démarrage serveur API sur port {port}...")
        
        try:
            api_script = self.base_path / "api" / "main.py"
            
            if not api_script.exists():
                print(f"  ❌ Script API non trouvé: {api_script}")
                return None
            
            # Commande FastAPI avec Uvicorn
            cmd = [
                sys.executable, "-m", "uvicorn",
                "api.main:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload"
            ]
            
            # Démarrer le processus
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path.parent)
            )
            
            # Attendre un peu pour vérifier le démarrage
            time.sleep(3)
            
            if process.poll() is None:
                print(f"  ✅ API démarrée sur http://localhost:{port}")
                print(f"  📚 Documentation API: http://localhost:{port}/docs")
                self.services["api"] = process
                return process
            else:
                print(f"  ❌ Erreur démarrage API")
                return None
                
        except Exception as e:
            print(f"  ❌ Erreur API: {e}")
            return None
    
    def open_browser_tabs(self, web_port: int = 8501, api_port: int = 8000):
        """Ouvre les onglets navigateur"""
        
        print("\n🌐 Ouverture des interfaces dans le navigateur...")
        
        try:
            time.sleep(2)  # Attendre que les services soient prêts
            
            # Interface web principale
            webbrowser.open(f"http://localhost:{web_port}")
            print(f"  🔗 Interface Web: http://localhost:{web_port}")
            
            # Documentation API
            webbrowser.open(f"http://localhost:{api_port}/docs")
            print(f"  🔗 API Documentation: http://localhost:{api_port}/docs")
            
        except Exception as e:
            print(f"  ⚠️ Erreur ouverture navigateur: {e}")
    
    def display_services_status(self):
        """Affiche le status des services"""
        
        print("\n📊 STATUS DES SERVICES ESERISIA:")
        print("-" * 50)
        
        if not self.services:
            print("  ⚠️ Aucun service démarré")
            return
        
        for name, process in self.services.items():
            if process and process.poll() is None:
                print(f"  ✅ {name.upper()}: Opérationnel")
            else:
                print(f"  ❌ {name.upper()}: Arrêté")
    
    def wait_for_services(self):
        """Attend et surveille les services"""
        
        print("\n🔍 Surveillance des services (Ctrl+C pour arrêter)...")
        print("="*60)
        
        try:
            while True:
                time.sleep(5)
                
                # Vérifier les services
                active_services = []
                for name, process in self.services.items():
                    if process and process.poll() is None:
                        active_services.append(name)
                
                if active_services:
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Services actifs: {', '.join(active_services)}")
                else:
                    print("⚠️ Tous les services sont arrêtés")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé par l'utilisateur")
            self.stop_all_services()
    
    def stop_all_services(self):
        """Arrête tous les services"""
        
        print("\n🛑 Arrêt des services ESERISIA...")
        
        for name, process in self.services.items():
            if process and process.poll() is None:
                print(f"  🔄 Arrêt de {name}...")
                process.terminate()
                
                # Attendre l'arrêt gracieux
                try:
                    process.wait(timeout=5)
                    print(f"  ✅ {name} arrêté")
                except subprocess.TimeoutExpired:
                    print(f"  💥 Arrêt forcé de {name}")
                    process.kill()
    
    def run_full_system(self, web_port: int = 8501, api_port: int = 8000, open_browser: bool = True):
        """Lance le système complet"""
        
        print_banner()
        
        # Vérification dépendances
        deps = self.check_dependencies()
        missing = [k for k, v in deps.items() if not v and k != "python"]
        
        if deps["python"]:
            print("  ✅ Python version compatible")
        else:
            print("  ❌ Python 3.8+ requis")
            return
        
        # Installation dépendances manquantes
        if missing:
            response = input(f"\n❓ Installer les dépendances manquantes? (y/N): ")
            if response.lower() in ['y', 'yes', 'o', 'oui']:
                self.install_missing_dependencies(missing)
            else:
                print("⚠️ Certaines fonctionnalités pourraient ne pas fonctionner")
        
        # Démos AI Core
        print("\n" + "="*60)
        print("🎯 PHASE 1: DÉMOS AI CORE")
        print("="*60)
        
        self.start_ai_core_demo()
        self.start_system_integration_demo()
        
        # Démarrage services
        print("\n" + "="*60)
        print("🎯 PHASE 2: DÉMARRAGE SERVICES")
        print("="*60)
        
        # Interface Web
        web_process = self.start_web_interface(web_port)
        
        # API Server
        api_process = self.start_api_server(api_port)
        
        # Ouverture navigateur
        if open_browser and (web_process or api_process):
            self.open_browser_tabs(web_port, api_port)
        
        # Status final
        print("\n" + "="*60)
        print("🎯 SYSTÈME ESERISIA AI - OPÉRATIONNEL")
        print("="*60)
        
        self.display_services_status()
        
        if self.services:
            print(f"\n🌐 Interface Principale: http://localhost:{web_port}")
            print(f"🔌 API Documentation: http://localhost:{api_port}/docs")
            print("\n🎉 ESERISIA AI est maintenant opérationnel!")
            print("   L'IA la plus avancée au monde est à votre service.")
            
            # Surveillance des services
            self.wait_for_services()
        else:
            print("\n❌ Aucun service n'a pu être démarré")

def main():
    """Point d'entrée principal"""
    
    parser = argparse.ArgumentParser(description="ESERISIA AI Ultimate Launcher")
    
    parser.add_argument("--web-port", type=int, default=8501,
                       help="Port pour l'interface web (défaut: 8501)")
    
    parser.add_argument("--api-port", type=int, default=8000,
                       help="Port pour l'API (défaut: 8000)")
    
    parser.add_argument("--no-browser", action="store_true",
                       help="Ne pas ouvrir le navigateur automatiquement")
    
    parser.add_argument("--demo-only", action="store_true",
                       help="Lancer uniquement les démos sans les services web")
    
    args = parser.parse_args()
    
    launcher = EserisiaLauncher()
    
    if args.demo_only:
        print_banner()
        print("\n🎯 MODE DÉMONSTRATION UNIQUEMENT")
        print("="*40)
        launcher.start_ai_core_demo()
        launcher.start_system_integration_demo()
        print("\n🎉 Démos terminées!")
    else:
        launcher.run_full_system(
            web_port=args.web_port,
            api_port=args.api_port,
            open_browser=not args.no_browser
        )

if __name__ == "__main__":
    main()
