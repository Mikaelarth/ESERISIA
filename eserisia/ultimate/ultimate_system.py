"""ESERISIA Ultimate integration system.

Production-oriented orchestrator that composes available subsystems
without requiring every optional module to be installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[ULTIMATE] %(asctime)s: %(message)s")
ultimate_logger = logging.getLogger("ESERISIA_ULTIMATE")

AI_CORE_AVAILABLE = False
QUANTUM_CORE_AVAILABLE = False
CONSCIOUSNESS_CORE_AVAILABLE = False
EVOLUTION_ENGINE_AVAILABLE = False

EserisiaResponse = Any
eserisia_ai = None
eserisia_quantum = None
eserisia_consciousness = None
eserisia_evolution = None

# NOTE:
# ai_core_live bootstraps heavyweight HF pipelines at import time.
# We keep it opt-in via env var to avoid side-effects in normal imports/tests.
if os.getenv("ESERISIA_ENABLE_AI_CORE", "0") == "1":
    try:
        from ..ai_core_live import EserisiaResponse, eserisia_ai

        AI_CORE_AVAILABLE = eserisia_ai is not None
    except Exception as exc:  # pragma: no cover - optional dependency surface
        ultimate_logger.warning("AI core unavailable: %s", exc)

try:
    from ..quantum.quantum_core import eserisia_quantum

    QUANTUM_CORE_AVAILABLE = eserisia_quantum is not None
except Exception as exc:  # pragma: no cover - optional dependency surface
    ultimate_logger.warning("Quantum core unavailable: %s", exc)

try:
    from ..consciousness.consciousness_core import eserisia_consciousness

    CONSCIOUSNESS_CORE_AVAILABLE = eserisia_consciousness is not None
except Exception as exc:  # pragma: no cover - optional dependency surface
    ultimate_logger.warning("Consciousness core unavailable: %s", exc)

try:
    from ..evolution.evolution_engine import eserisia_evolution

    EVOLUTION_ENGINE_AVAILABLE = eserisia_evolution is not None
except Exception as exc:  # pragma: no cover - optional dependency surface
    ultimate_logger.warning("Evolution engine unavailable: %s", exc)


@dataclass
class UltimateResponse:
    content: str
    confidence: float
    processing_time: float
    consciousness_level: float
    quantum_advantage: bool
    distributed_processing: bool
    creative_insights: int
    evolution_applied: bool
    ultimate_intelligence: float
    transcendence_factor: float
    evolutionary_generation: int = 0
    fitness_score: float = 0.0


class EserisiaUltimateSystem:
    """Compose available ESERISIA modules with defensive runtime behavior."""

    def __init__(self) -> None:
        self.version = "ultimate-2.0"
        self.start_time = datetime.now()
        self.integration_status = {
            "ai_core": AI_CORE_AVAILABLE,
            "quantum_core": QUANTUM_CORE_AVAILABLE,
            "consciousness_core": CONSCIOUSNESS_CORE_AVAILABLE,
            "evolution_engine": EVOLUTION_ENGINE_AVAILABLE,
        }
        self.strict_mode = os.getenv("ESERISIA_STRICT_MODE", "1") != "0"
        self.operations_count = 0

    async def ultimate_process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_consciousness: bool = True,
        use_quantum: bool = True,
        use_evolution: bool = True,
        distributed: bool = False,
    ) -> UltimateResponse:
        start = datetime.now()
        context = context or {}
        self.operations_count += 1

        confidence = 0.50
        content_chunks = []
        consciousness_level = 0.0
        creative_insights = 0
        quantum_advantage = False
        evolution_applied = False
        evolutionary_generation = 0
        fitness_score = 0.0

        try:
            if self.integration_status["ai_core"]:
                ai_response: EserisiaResponse = await eserisia_ai.process_request(
                    query, request_type="analysis", context=context
                )
                content_chunks.append(ai_response.content)
                confidence = max(confidence, float(ai_response.confidence))
            else:
                content_chunks.append(self._fallback_text(query))

            if self.integration_status["consciousness_core"] and use_consciousness:
                result = await eserisia_consciousness.conscious_reasoning(query, context)
                consciousness_level = float(result.get("consciousness_level", 0.0))
                creative_insights = int(result.get("creative_insights_used", 0))
                content_chunks.append(result.get("response", ""))

            if self.integration_status["quantum_core"] and use_quantum:
                qres = await eserisia_quantum.quantum_neural_optimization(
                    {"weights": [0.1, 0.2], "bias": [0.0]}
                )
                quantum_advantage = bool(getattr(qres, "quantum_advantage", False))

            if self.integration_status["evolution_engine"] and use_evolution:
                metrics = await eserisia_evolution.evolve_generation()
                fitness_score = float(metrics.fitness_score)
                evolutionary_generation = int(metrics.generation)
                evolution_applied = True
                confidence = min(0.999, confidence + max(0.0, float(metrics.intelligence_gain)))

            processing_time = (datetime.now() - start).total_seconds()
            ultimate_intelligence = self._compute_intelligence(
                confidence, consciousness_level, quantum_advantage
            )
            transcendence_factor = self._compute_transcendence(
                ultimate_intelligence, creative_insights, quantum_advantage
            )

            return UltimateResponse(
                content="\n\n".join([c for c in content_chunks if c]).strip(),
                confidence=confidence,
                processing_time=processing_time,
                consciousness_level=consciousness_level,
                quantum_advantage=quantum_advantage,
                distributed_processing=distributed,
                creative_insights=creative_insights,
                evolution_applied=evolution_applied,
                ultimate_intelligence=ultimate_intelligence,
                transcendence_factor=transcendence_factor,
                evolutionary_generation=evolutionary_generation,
                fitness_score=fitness_score,
            )
        except Exception as exc:
            ultimate_logger.exception("ultimate_process failed")
            return self._error_response(query, start, str(exc))

    def get_ultimate_status(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "version": self.version,
            "strict_mode": self.strict_mode,
            "uptime_seconds": uptime,
            "operations_count": self.operations_count,
            "integration_status": self.integration_status,
        }

    def _compute_intelligence(
        self, confidence: float, consciousness: float, quantum_advantage: bool
    ) -> float:
        score = confidence * 0.6 + consciousness * 0.3 + (0.1 if quantum_advantage else 0.0)
        return min(1.0, max(0.0, score))

    def _compute_transcendence(
        self, intelligence: float, insights: int, quantum: bool
    ) -> float:
        insights_boost = min(0.3, insights / 20.0)
        quantum_boost = 0.1 if quantum else 0.0
        return min(1.0, intelligence * 0.6 + insights_boost + quantum_boost)

    def _fallback_text(self, query: str) -> str:
        return (
            "Mode fallback actif: aucun module IA principal disponible. "
            f"Requête reçue: {query}"
        )

    def _error_response(self, query: str, start: datetime, message: str) -> UltimateResponse:
        return UltimateResponse(
            content=f"Erreur de traitement pour '{query}': {message}",
            confidence=0.0,
            processing_time=(datetime.now() - start).total_seconds(),
            consciousness_level=0.0,
            quantum_advantage=False,
            distributed_processing=False,
            creative_insights=0,
            evolution_applied=False,
            ultimate_intelligence=0.0,
            transcendence_factor=0.0,
        )


try:
    eserisia_ultimate = EserisiaUltimateSystem()
except Exception as exc:  # pragma: no cover
    ultimate_logger.error("Ultimate system bootstrap failed: %s", exc)
    eserisia_ultimate = None


async def ask_ultimate_eserisia(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    full_power: bool = True,
) -> str:
    if eserisia_ultimate is None:
        return "Système ultimate indisponible"

    response = await eserisia_ultimate.ultimate_process(
        query,
        context=context,
        use_consciousness=full_power,
        use_quantum=full_power,
        use_evolution=full_power,
    )
    return response.content


async def ultimate_demo() -> None:
    if eserisia_ultimate is None:
        print("Système ultimate indisponible")
        return

    response = await eserisia_ultimate.ultimate_process("diagnostic système")
    print(response.content)
    print(eserisia_ultimate.get_ultimate_status())


if __name__ == "__main__":
    asyncio.run(ultimate_demo())
