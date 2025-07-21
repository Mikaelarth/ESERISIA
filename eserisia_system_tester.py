"""
ESERISIA AI - TESTS COMPLETS DU SYSTÈME
======================================
Validation complète de toutes les fonctionnalités ESERISIA AI
Tests d'intégration et de performance
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Import des composants ESERISIA
try:
    from eserisia.ai_core_live import eserisia_ai, ask_eserisia, get_eserisia_status
    AI_CORE_AVAILABLE = True
except ImportError as e:
    AI_CORE_AVAILABLE = False
    print(f"⚠️ AI Core non disponible: {e}")

try:
    from eserisia.ide_engine import EserisiaIDE
    IDE_AVAILABLE = True
except ImportError as e:
    IDE_AVAILABLE = False
    print(f"⚠️ IDE Engine non disponible: {e}")

class EserisiaSystemTester:
    """Testeur complet pour le système ESERISIA AI"""
    
    def __init__(self):
        """Initialise le testeur"""
        self.version = "2.0.0-TESTING"
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "performance_metrics": {
                "total_time": 0.0,
                "average_response_time": 0.0,
                "fastest_response": float('inf'),
                "slowest_response": 0.0
            },
            "components_tested": []
        }
        
        print("🧪 ESERISIA AI - TESTEUR SYSTÈME COMPLET")
        print("=" * 60)
        print(f"Version: {self.version}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
    
    def add_test_result(self, test_name: str, success: bool, duration: float, 
                       details: Dict[str, Any] = None):
        """Ajoute un résultat de test"""
        
        self.results["total_tests"] += 1
        
        if success:
            self.results["passed_tests"] += 1
        else:
            self.results["failed_tests"] += 1
        
        # Métriques de performance
        self.results["performance_metrics"]["total_time"] += duration
        self.results["performance_metrics"]["fastest_response"] = min(
            self.results["performance_metrics"]["fastest_response"], duration
        )
        self.results["performance_metrics"]["slowest_response"] = max(
            self.results["performance_metrics"]["slowest_response"], duration
        )
        
        if self.results["total_tests"] > 0:
            self.results["performance_metrics"]["average_response_time"] = (
                self.results["performance_metrics"]["total_time"] / 
                self.results["total_tests"]
            )
        
        # Enregistrer résultat
        result = {
            "test_name": test_name,
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.results["test_results"].append(result)
        
        # Affichage
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.3f}s)")
        
        if not success and details and "error" in details:
            print(f"    Error: {details['error']}")
    
    async def test_ai_core_basic(self) -> bool:
        """Test basique de l'AI Core"""
        
        if not AI_CORE_AVAILABLE:
            self.add_test_result("AI Core Basic", False, 0.0, 
                               {"error": "AI Core non disponible"})
            return False
        
        start_time = time.time()
        
        try:
            # Test d'initialisation
            status = get_eserisia_status()
            
            if "OPERATIONAL" in status.get("status", ""):
                duration = time.time() - start_time
                self.add_test_result("AI Core Basic", True, duration, {
                    "intelligence_level": status.get("intelligence_level", 0),
                    "precision": status.get("precision_rate", "0%"),
                    "version": status.get("version", "Unknown")
                })
                return True
            else:
                duration = time.time() - start_time
                self.add_test_result("AI Core Basic", False, duration, 
                                   {"error": "Status non opérationnel"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_test_result("AI Core Basic", False, duration, 
                               {"error": str(e)})
            return False
    
    async def test_ai_core_text_generation(self) -> bool:
        """Test de génération de texte"""
        
        if not AI_CORE_AVAILABLE:
            self.add_test_result("AI Text Generation", False, 0.0, 
                               {"error": "AI Core non disponible"})
            return False
        
        start_time = time.time()
        
        try:
            response = await ask_eserisia(
                "Décris en une phrase les capacités d'ESERISIA AI", 
                "general"
            )
            
            duration = time.time() - start_time
            
            if response and len(response) > 10:
                self.add_test_result("AI Text Generation", True, duration, {
                    "response_length": len(response),
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                })
                return True
            else:
                self.add_test_result("AI Text Generation", False, duration, 
                                   {"error": "Réponse vide ou trop courte"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_test_result("AI Text Generation", False, duration, 
                               {"error": str(e)})
            return False
    
    async def test_ai_core_code_generation(self) -> bool:
        """Test de génération de code"""
        
        if not AI_CORE_AVAILABLE:
            self.add_test_result("AI Code Generation", False, 0.0, 
                               {"error": "AI Core non disponible"})
            return False
        
        start_time = time.time()
        
        try:
            response = await ask_eserisia(
                "Génère une fonction Python simple", 
                "code"
            )
            
            duration = time.time() - start_time
            
            # Vérifier que c'est du code Python
            if response and ("def " in response or "class " in response):
                self.add_test_result("AI Code Generation", True, duration, {
                    "code_length": len(response),
                    "contains_function": "def " in response,
                    "contains_class": "class " in response
                })
                return True
            else:
                self.add_test_result("AI Code Generation", False, duration, 
                                   {"error": "Code généré non valide"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_test_result("AI Code Generation", False, duration, 
                               {"error": str(e)})
            return False
    
    async def test_ai_core_analysis(self) -> bool:
        """Test d'analyse de contenu"""
        
        if not AI_CORE_AVAILABLE:
            self.add_test_result("AI Content Analysis", False, 0.0, 
                               {"error": "AI Core non disponible"})
            return False
        
        start_time = time.time()
        
        try:
            response = await ask_eserisia(
                "Analyse ce texte: L'intelligence artificielle révolutionne le monde", 
                "analysis"
            )
            
            duration = time.time() - start_time
            
            if response and ("ANALYSE" in response or "analyse" in response):
                self.add_test_result("AI Content Analysis", True, duration, {
                    "analysis_length": len(response),
                    "contains_metrics": "%" in response
                })
                return True
            else:
                self.add_test_result("AI Content Analysis", False, duration, 
                                   {"error": "Analyse non générée"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_test_result("AI Content Analysis", False, duration, 
                               {"error": str(e)})
            return False
    
    async def test_ide_engine_basic(self) -> bool:
        """Test basique de l'IDE Engine"""
        
        if not IDE_AVAILABLE:
            self.add_test_result("IDE Engine Basic", False, 0.0, 
                               {"error": "IDE Engine non disponible"})
            return False
        
        start_time = time.time()
        
        try:
            ide = EserisiaIDE(".")
            status = ide.get_ide_status()
            
            duration = time.time() - start_time
            
            if status and "ESERISIA AI" in status.get("system", ""):
                self.add_test_result("IDE Engine Basic", True, duration, {
                    "precision": status.get("precision", "Unknown"),
                    "languages_supported": len(status.get("supported_languages", [])),
                    "capabilities": len(status.get("capabilities", []))
                })
                return True
            else:
                self.add_test_result("IDE Engine Basic", False, duration, 
                                   {"error": "Status IDE non valide"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_test_result("IDE Engine Basic", False, duration, 
                               {"error": str(e)})
            return False
    
    async def test_performance_stress(self, iterations: int = 10) -> bool:
        """Test de performance sous charge"""
        
        if not AI_CORE_AVAILABLE:
            self.add_test_result("Performance Stress Test", False, 0.0, 
                               {"error": "AI Core non disponible"})
            return False
        
        start_time = time.time()
        
        try:
            successful_requests = 0
            response_times = []
            
            for i in range(iterations):
                request_start = time.time()
                
                try:
                    response = await ask_eserisia(f"Test performance {i+1}", "general")
                    
                    if response and len(response) > 0:
                        successful_requests += 1
                        response_times.append(time.time() - request_start)
                
                except Exception:
                    pass
            
            duration = time.time() - start_time
            success_rate = (successful_requests / iterations) * 100
            
            if success_rate >= 80:  # 80% de succès minimum
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                requests_per_second = successful_requests / duration if duration > 0 else 0
                
                self.add_test_result("Performance Stress Test", True, duration, {
                    "iterations": iterations,
                    "successful_requests": successful_requests,
                    "success_rate": f"{success_rate:.1f}%",
                    "average_response_time": f"{avg_response_time:.3f}s",
                    "requests_per_second": f"{requests_per_second:.2f}"
                })
                return True
            else:
                self.add_test_result("Performance Stress Test", False, duration, 
                                   {"error": f"Taux de succès trop faible: {success_rate:.1f}%"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_test_result("Performance Stress Test", False, duration, 
                               {"error": str(e)})
            return False
    
    async def test_system_integration(self) -> bool:
        """Test d'intégration système"""
        
        start_time = time.time()
        
        try:
            # Test d'intégration entre AI Core et IDE
            integration_success = True
            
            # Vérifier que les composants peuvent être importés
            if AI_CORE_AVAILABLE:
                integration_success = True
            else:
                integration_success = False
            
            if IDE_AVAILABLE:
                integration_success = integration_success and True
            else:
                integration_success = False
            
            duration = time.time() - start_time
            
            components = []
            if AI_CORE_AVAILABLE:
                components.append("AI Core")
            if IDE_AVAILABLE:
                components.append("IDE Engine")
            
            if integration_success:
                self.add_test_result("System Integration", True, duration, {
                    "components_available": components,
                    "integration_level": "Basique"
                })
                return True
            else:
                self.add_test_result("System Integration", False, duration, 
                                   {"error": "Composants manquants", "available": components})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_test_result("System Integration", False, duration, 
                               {"error": str(e)})
            return False
    
    def display_results_summary(self):
        """Affiche le résumé des résultats"""
        
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS DES TESTS ESERISIA AI")
        print("=" * 60)
        
        # Statistiques générales
        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"📈 Tests totaux: {total}")
        print(f"✅ Tests réussis: {passed}")
        print(f"❌ Tests échoués: {failed}")
        print(f"🎯 Taux de succès: {success_rate:.1f}%")
        
        # Métriques de performance
        perf = self.results["performance_metrics"]
        print(f"\n⏱️ Performance:")
        print(f"   Temps total: {perf['total_time']:.3f}s")
        print(f"   Temps moyen: {perf['average_response_time']:.3f}s")
        print(f"   Plus rapide: {perf['fastest_response']:.3f}s")
        print(f"   Plus lent: {perf['slowest_response']:.3f}s")
        
        # Détails des échecs
        failed_tests = [r for r in self.results["test_results"] if not r["success"]]
        if failed_tests:
            print(f"\n❌ Échecs détaillés:")
            for test in failed_tests:
                print(f"   • {test['test_name']}: {test['details'].get('error', 'Erreur inconnue')}")
        
        # Status final
        print("\n" + "=" * 60)
        
        if success_rate >= 90:
            print("🎉 SYSTÈME ESERISIA AI - EXCELLENT! 🎉")
        elif success_rate >= 70:
            print("✅ SYSTÈME ESERISIA AI - BON")
        elif success_rate >= 50:
            print("⚠️ SYSTÈME ESERISIA AI - MOYEN")
        else:
            print("❌ SYSTÈME ESERISIA AI - PROBLÈME")
        
        print("=" * 60)
    
    def save_results_to_file(self, filename: str = None):
        """Sauvegarde les résultats dans un fichier"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eserisia_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Résultats sauvegardés dans: {filename}")
        
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    async def run_all_tests(self):
        """Lance tous les tests"""
        
        print("🚀 LANCEMENT DE TOUS LES TESTS ESERISIA AI")
        print("-" * 60)
        
        # Tests AI Core
        if AI_CORE_AVAILABLE:
            self.results["components_tested"].append("AI Core")
            
            await self.test_ai_core_basic()
            await self.test_ai_core_text_generation()
            await self.test_ai_core_code_generation()
            await self.test_ai_core_analysis()
            
            # Test de stress
            await self.test_performance_stress(5)  # 5 itérations pour être rapide
        
        # Tests IDE Engine
        if IDE_AVAILABLE:
            self.results["components_tested"].append("IDE Engine")
            await self.test_ide_engine_basic()
        
        # Test d'intégration
        await self.test_system_integration()
        
        # Résultats
        self.display_results_summary()
        self.save_results_to_file()

# Fonction principale pour lancer les tests
async def main():
    """Lance les tests complets"""
    
    tester = EserisiaSystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
