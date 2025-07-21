"""
ESERISIA AI - Base de Données Évolutive PostgreSQL Pure
======================================================
Architecture ultra-performante pour IA évolutive (PostgreSQL uniquement)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
import hashlib
import statistics
from pathlib import Path
import os

# Base de données PostgreSQL
import asyncpg

logger = logging.getLogger(__name__)

@dataclass
class LearningEvent:
    """Événement d'apprentissage ESERISIA"""
    event_type: str
    context: Dict[str, Any]
    timestamp: datetime = None
    user_feedback: Optional[int] = None
    success: bool = True
    improvement_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    project_hash: Optional[str] = None
    file_path: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class CodePattern:
    """Pattern de code appris"""
    pattern_hash: str
    pattern_type: str
    content: str
    language: Optional[str] = None
    frequency: int = 1
    success_rate: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class ProjectInsight:
    """Insight sur un projet"""
    project_hash: str
    insight_type: str
    content: Dict[str, Any]
    confidence: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

class EserisiaDatabase:
    """
    Base de données évolutive ultra-avancée ESERISIA AI
    Architecture: PostgreSQL pour performance et évolutivité maximale
    """
    
    def __init__(self):
        self.pg_pool = None
        self.stats = {}
        self.intelligence_level = 1.0
        self.evolution_history = []
        
        # Configuration PostgreSQL
        self.pg_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'yF_FAMILLEY1983'),
            'database': os.getenv('POSTGRES_DB', 'eserisia_ai'),
            'min_size': 5,
            'max_size': 20,
            'command_timeout': 60
        }
        
        logger.info("🚀 ESERISIA Database initialisé (PostgreSQL)")
    
    async def initialize(self) -> bool:
        """Initialisation complète base de données"""
        try:
            await self._setup_postgresql()
            await self._load_intelligence_level()
            await self._load_stats()
            
            logger.info("✅ Base évolutive ESERISIA opérationnelle")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation database: {e}")
            return False
    
    async def _setup_postgresql(self):
        """Configuration PostgreSQL avec pool de connexions"""
        try:
            # Test et création de la base si nécessaire
            try:
                # Connexion admin pour créer DB si nécessaire
                admin_conn = await asyncpg.connect(
                    host=self.pg_config['host'],
                    port=self.pg_config['port'],
                    user=self.pg_config['user'],
                    password=self.pg_config['password'],
                    database='postgres'
                )
                
                # Créer base si n'existe pas
                try:
                    await admin_conn.execute(f"CREATE DATABASE {self.pg_config['database']}")
                    logger.info(f"✅ Base '{self.pg_config['database']}' créée")
                except asyncpg.DuplicateDatabaseError:
                    pass  # Base existe déjà
                
                await admin_conn.close()
                
            except Exception as e:
                logger.warning(f"⚠️ Utilisation base existante: {e}")
            
            # Créer pool de connexions
            self.pg_pool = await asyncpg.create_pool(
                host=self.pg_config['host'],
                port=self.pg_config['port'],
                user=self.pg_config['user'],
                password=self.pg_config['password'],
                database=self.pg_config['database'],
                min_size=self.pg_config['min_size'],
                max_size=self.pg_config['max_size'],
                command_timeout=self.pg_config['command_timeout']
            )
            
            # Créer tables
            await self._create_tables()
            
            logger.info("✅ PostgreSQL configuré avec pool de connexions")
            
        except Exception as e:
            logger.error(f"❌ Erreur PostgreSQL: {e}")
            raise
    
    async def _create_tables(self):
        """Création tables optimisées PostgreSQL"""
        tables = [
            # Événements apprentissage
            """
            CREATE TABLE IF NOT EXISTS learning_events (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                event_type VARCHAR(50) NOT NULL,
                context JSONB NOT NULL,
                user_feedback INTEGER,
                success BOOLEAN DEFAULT TRUE,
                improvement_score FLOAT,
                metadata JSONB,
                project_hash VARCHAR(64),
                file_path TEXT
            )
            """,
            
            # Patterns de code
            """
            CREATE TABLE IF NOT EXISTS code_patterns (
                id SERIAL PRIMARY KEY,
                pattern_hash VARCHAR(64) UNIQUE NOT NULL,
                pattern_type VARCHAR(50) NOT NULL,
                language VARCHAR(30),
                content TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                success_rate FLOAT DEFAULT 0.0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB
            )
            """,
            
            # Insights projet
            """
            CREATE TABLE IF NOT EXISTS project_insights (
                id SERIAL PRIMARY KEY,
                project_hash VARCHAR(64) NOT NULL,
                insight_type VARCHAR(50) NOT NULL,
                content JSONB NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            
            # Évolution IA
            """
            CREATE TABLE IF NOT EXISTS ai_evolution (
                id SERIAL PRIMARY KEY,
                version VARCHAR(20) NOT NULL,
                optimization_type VARCHAR(50) NOT NULL,
                performance_before JSONB,
                performance_after JSONB,
                improvement_metrics JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                active BOOLEAN DEFAULT TRUE
            )
            """,
            
            # Sessions d'apprentissage
            """
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id SERIAL PRIMARY KEY,
                session_hash VARCHAR(64) UNIQUE NOT NULL,
                started_at TIMESTAMPTZ DEFAULT NOW(),
                ended_at TIMESTAMPTZ,
                events_count INTEGER DEFAULT 0,
                improvements_made INTEGER DEFAULT 0,
                intelligence_gain FLOAT DEFAULT 0.0,
                metadata JSONB
            )
            """
        ]
        
        # Index pour performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_learning_events_timestamp ON learning_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_learning_events_type ON learning_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_learning_events_project ON learning_events(project_hash)",
            "CREATE INDEX IF NOT EXISTS idx_code_patterns_hash ON code_patterns(pattern_hash)",
            "CREATE INDEX IF NOT EXISTS idx_code_patterns_type ON code_patterns(pattern_type)",
            "CREATE INDEX IF NOT EXISTS idx_code_patterns_lang ON code_patterns(language)",
            "CREATE INDEX IF NOT EXISTS idx_project_insights_hash ON project_insights(project_hash)",
            "CREATE INDEX IF NOT EXISTS idx_project_insights_type ON project_insights(insight_type)",
            "CREATE INDEX IF NOT EXISTS idx_ai_evolution_type ON ai_evolution(optimization_type)",
            "CREATE INDEX IF NOT EXISTS idx_learning_sessions_hash ON learning_sessions(session_hash)"
        ]
        
        async with self.pg_pool.acquire() as conn:
            # Créer tables
            for table_sql in tables:
                await conn.execute(table_sql)
            
            # Créer index
            for index_sql in indexes:
                await conn.execute(index_sql)
        
        logger.info("✅ Tables PostgreSQL créées avec index optimisés")
    
    async def record_learning_event(self, event: LearningEvent) -> bool:
        """Enregistre événement d'apprentissage"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO learning_events (
                        timestamp, event_type, context, user_feedback,
                        success, improvement_score, metadata, project_hash, file_path
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    event.timestamp, event.event_type, json.dumps(event.context),
                    event.user_feedback, event.success, event.improvement_score,
                    json.dumps(event.metadata) if event.metadata else None,
                    event.project_hash, event.file_path
                )
            
            # Mise à jour stats
            self.stats["total_analyses"] = self.stats.get("total_analyses", 0) + 1
            if event.success and event.event_type == "optimization":
                self.stats["successful_optimizations"] = self.stats.get("successful_optimizations", 0) + 1
            
            logger.debug(f"✅ Événement enregistré: {event.event_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur enregistrement: {e}")
            return False
    
    async def learn_code_pattern(self, pattern: CodePattern) -> bool:
        """Apprend un nouveau pattern de code"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Vérifier si pattern existe
                existing = await conn.fetchrow(
                    "SELECT id, frequency FROM code_patterns WHERE pattern_hash = $1",
                    pattern.pattern_hash
                )
                
                if existing:
                    # Mettre à jour fréquence
                    await conn.execute(
                        """
                        UPDATE code_patterns 
                        SET frequency = frequency + 1, updated_at = NOW()
                        WHERE pattern_hash = $1
                        """,
                        pattern.pattern_hash
                    )
                else:
                    # Nouveau pattern
                    await conn.execute(
                        """
                        INSERT INTO code_patterns (
                            pattern_hash, pattern_type, language, content,
                            frequency, success_rate, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        pattern.pattern_hash, pattern.pattern_type, pattern.language,
                        pattern.content, pattern.frequency, pattern.success_rate,
                        json.dumps(pattern.metadata) if pattern.metadata else None
                    )
            
            # Stats
            if not existing:
                self.stats["patterns_learned"] = self.stats.get("patterns_learned", 0) + 1
            
            logger.debug(f"✅ Pattern appris: {pattern.pattern_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur apprentissage pattern: {e}")
            return False
    
    async def get_similar_patterns(self, pattern_type: str, language: Optional[str] = None) -> List[CodePattern]:
        """Récupère patterns similaires"""
        try:
            async with self.pg_pool.acquire() as conn:
                if language:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM code_patterns 
                        WHERE pattern_type = $1 AND language = $2
                        ORDER BY frequency DESC, success_rate DESC
                        LIMIT 10
                        """,
                        pattern_type, language
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM code_patterns 
                        WHERE pattern_type = $1
                        ORDER BY frequency DESC, success_rate DESC
                        LIMIT 10
                        """,
                        pattern_type
                    )
                
                patterns = []
                for row in rows:
                    patterns.append(CodePattern(
                        pattern_hash=row['pattern_hash'],
                        pattern_type=row['pattern_type'],
                        content=row['content'],
                        language=row['language'],
                        frequency=row['frequency'],
                        success_rate=row['success_rate'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    ))
                
                return patterns
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération patterns: {e}")
            return []
    
    async def evolve_intelligence(self, optimization_data: Dict[str, Any]) -> float:
        """Évolution de l'intelligence IA"""
        try:
            # Calculer amélioration
            old_level = self.intelligence_level
            improvement = optimization_data.get("improvement_score", 0.0)
            
            # Formule évolution ultra-avancée
            self.intelligence_level += improvement * 0.001  # Croissance contrôlée
            self.intelligence_level = min(self.intelligence_level, 10.0)  # Cap à 10.0
            
            # Enregistrer évolution
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO ai_evolution (
                        version, optimization_type, performance_before,
                        performance_after, improvement_metrics
                    ) VALUES ($1, $2, $3, $4, $5)
                    """,
                    f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    optimization_data.get("type", "general"),
                    json.dumps({"intelligence_level": old_level}),
                    json.dumps({"intelligence_level": self.intelligence_level}),
                    json.dumps(optimization_data)
                )
            
            self.evolution_history.append({
                "timestamp": datetime.utcnow(),
                "old_level": old_level,
                "new_level": self.intelligence_level,
                "improvement": self.intelligence_level - old_level
            })
            
            logger.info(f"🧠 Intelligence évoluée: {old_level:.3f} → {self.intelligence_level:.3f}")
            return self.intelligence_level
            
        except Exception as e:
            logger.error(f"❌ Erreur évolution: {e}")
            return self.intelligence_level
    
    async def get_learning_analytics(self) -> Dict[str, Any]:
        """Analytics complets d'apprentissage"""
        try:
            analytics = {}
            
            async with self.pg_pool.acquire() as conn:
                # Statistiques générales
                analytics["events_total"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM learning_events"
                ) or 0
                
                analytics["events_today"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM learning_events WHERE timestamp >= NOW() - INTERVAL '1 day'"
                ) or 0
                
                success_rate = await conn.fetchval(
                    "SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) FROM learning_events"
                )
                analytics["success_rate"] = float(success_rate) if success_rate else 0.0
                
                # Patterns par langage
                pattern_stats = await conn.fetch(
                    """
                    SELECT language, COUNT(*) as count, AVG(success_rate) as avg_success
                    FROM code_patterns 
                    WHERE language IS NOT NULL
                    GROUP BY language 
                    ORDER BY count DESC
                    """
                )
                
                analytics["patterns_by_language"] = [
                    {
                        "language": row["language"], 
                        "count": row["count"], 
                        "avg_success": float(row["avg_success"]) if row["avg_success"] else 0.0
                    }
                    for row in pattern_stats
                ]
                
                # Évolution intelligence
                latest_evolution = await conn.fetchrow(
                    "SELECT improvement_metrics FROM ai_evolution ORDER BY created_at DESC LIMIT 1"
                )
                
                analytics["intelligence"] = {
                    "current_level": self.intelligence_level,
                    "evolution_count": len(self.evolution_history),
                    "latest_improvement": json.loads(latest_evolution["improvement_metrics"]) if latest_evolution and latest_evolution["improvement_metrics"] else None
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Erreur analytics: {e}")
            return {"error": str(e)}
    
    async def get_intelligence_status(self) -> Dict[str, Any]:
        """Status complet de l'intelligence évolutive"""
        try:
            analytics = await self.get_learning_analytics()
            
            status = {
                "intelligence_level": self.intelligence_level,
                "level_description": self._get_intelligence_description(),
                "learning_stats": {
                    "total_events": analytics.get("events_total", 0),
                    "events_today": analytics.get("events_today", 0),
                    "success_rate": analytics.get("success_rate", 0.0),
                    "patterns_learned": analytics.get("patterns_by_language", [])
                },
                "evolution_metrics": {
                    "evolution_count": len(self.evolution_history),
                    "recent_improvements": self.evolution_history[-5:] if self.evolution_history else [],
                    "avg_improvement": statistics.mean([h["improvement"] for h in self.evolution_history]) if self.evolution_history else 0.0
                },
                "capabilities": self._get_current_capabilities(),
                "recommendations": self._get_evolution_recommendations()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Erreur status intelligence: {e}")
            return {"error": str(e)}
    
    def _get_intelligence_description(self) -> str:
        """Description du niveau d'intelligence actuel"""
        if self.intelligence_level < 1.5:
            return "Débutant - Apprentissage des bases"
        elif self.intelligence_level < 2.5:
            return "Intermédiaire - Reconnaissance de patterns"
        elif self.intelligence_level < 4.0:
            return "Avancé - Optimisation intelligente"
        elif self.intelligence_level < 6.0:
            return "Expert - Compréhension contextuelle"
        elif self.intelligence_level < 8.0:
            return "Maître - Innovation autonome"
        else:
            return "Ultra-Avancé - IA créative et évolutive"
    
    def _get_current_capabilities(self) -> List[str]:
        """Capacités actuelles basées sur le niveau d'intelligence"""
        base_capabilities = [
            "Analyse de code multi-langage",
            "Détection de patterns",
            "Suggestions d'amélioration"
        ]
        
        if self.intelligence_level >= 2.0:
            base_capabilities.extend([
                "Optimisation automatique",
                "Apprentissage des préférences utilisateur"
            ])
        
        if self.intelligence_level >= 3.0:
            base_capabilities.extend([
                "Génération de code contextuelle",
                "Architecture pattern recognition"
            ])
        
        if self.intelligence_level >= 4.0:
            base_capabilities.extend([
                "Refactorisation intelligente",
                "Tests automatiques"
            ])
        
        if self.intelligence_level >= 5.0:
            base_capabilities.extend([
                "Innovation de solutions",
                "Adaptation aux nouveaux frameworks"
            ])
        
        return base_capabilities
    
    def _get_evolution_recommendations(self) -> List[str]:
        """Recommandations pour l'évolution"""
        recommendations = []
        
        if self.stats.get("total_analyses", 0) < 50:
            recommendations.append("Analyser plus de fichiers pour accélérer l'apprentissage")
        
        if self.stats.get("patterns_learned", 0) < 20:
            recommendations.append("Explorer différents langages pour diversifier les patterns")
        
        if self.stats.get("successful_optimizations", 0) < 10:
            recommendations.append("Accepter plus de suggestions pour améliorer les algorithmes")
        
        if len(self.evolution_history) < 5:
            recommendations.append("Utiliser 'evolve' régulièrement pour progresser")
        
        return recommendations or ["Continuez à utiliser ESERISIA pour une évolution optimale"]
    
    async def _load_intelligence_level(self):
        """Charge niveau d'intelligence"""
        try:
            async with self.pg_pool.acquire() as conn:
                latest = await conn.fetchrow(
                    "SELECT improvement_metrics FROM ai_evolution ORDER BY created_at DESC LIMIT 1"
                )
                
                if latest and latest["improvement_metrics"]:
                    metrics = json.loads(latest["improvement_metrics"])
                    self.intelligence_level = metrics.get("intelligence_level", 1.0)
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement intelligence: {e}")
            self.intelligence_level = 1.0
    
    async def _load_stats(self):
        """Charge statistiques"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Événements totaux
                self.stats["total_analyses"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM learning_events"
                ) or 0
                
                # Optimisations réussies
                self.stats["successful_optimizations"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM learning_events WHERE success=true AND event_type='optimization'"
                ) or 0
                
                # Patterns appris
                self.stats["patterns_learned"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM code_patterns"
                ) or 0
                
                # Score amélioration moyen
                avg_score = await conn.fetchval(
                    "SELECT AVG(improvement_score) FROM learning_events WHERE improvement_score IS NOT NULL"
                )
                self.stats["average_improvement"] = float(avg_score) if avg_score else 0.0
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement stats: {e}")
            self.stats = {
                "total_analyses": 0,
                "successful_optimizations": 0,
                "patterns_learned": 0,
                "average_improvement": 0.0
            }
    
    async def close(self):
        """Fermeture propre"""
        if self.pg_pool:
            await self.pg_pool.close()
        
        logger.info("✅ Base de données fermée proprement")

# Instance globale
eserisia_db = EserisiaDatabase()

# Fonctions utilitaires
async def record_analysis_event(event_type: str, context: Dict[str, Any], 
                               success: bool = True, improvement_score: Optional[float] = None,
                               project_hash: Optional[str] = None, file_path: Optional[str] = None) -> bool:
    """Enregistre événement d'analyse"""
    event = LearningEvent(
        event_type=event_type,
        context=context,
        success=success,
        improvement_score=improvement_score,
        project_hash=project_hash,
        file_path=file_path
    )
    return await eserisia_db.record_learning_event(event)

async def learn_from_user_feedback(event_type: str, context: Dict[str, Any], 
                                 feedback_score: int) -> bool:
    """Apprentissage depuis feedback utilisateur"""
    event = LearningEvent(
        event_type=event_type,
        context=context,
        user_feedback=feedback_score,
        success=feedback_score > 0
    )
    return await eserisia_db.record_learning_event(event)

async def evolve_eserisia_intelligence(optimization_results: Dict[str, Any]) -> float:
    """Évolution intelligence ESERISIA"""
    return await eserisia_db.evolve_intelligence(optimization_results)
