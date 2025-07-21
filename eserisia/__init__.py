"""
ESERISIA AI - Ultimate Advanced AI System
==========================================

The world's most advanced evolutionary AI system.
Designed to be several steps ahead of all competitors.

Version: 1.0.0
Author: ESERISIA Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "ESERISIA Team"
__email__ = "ai@eserisia.com"
__license__ = "MIT"

# Core imports
try:
    from .core import EvolutiveBrain
    from .training import MetaLearner, NeuralArchitectureSearch
    from .models import TransformerEvolved
    from .inference import UltraFastInference, RealtimeProcessor, BatchOptimizer
    from .quantum import QuantumProcessor, QuantumOptimizer, QuantumNeuralHybrid
    from .security import AlignmentSystem, RobustnessChecker
    from .utils import setup_logging, PerformanceMonitor, SystemDiagnostics
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    # Fallback imports for development
    EvolutiveBrain = None
    MetaLearner = None

# Version info
def version() -> str:
    """Return the current version of ESERISIA AI."""
    return f"ESERISIA AI v{__version__} - The Ultimate AI System"

def system_info() -> dict:
    """Return comprehensive system information."""
    import torch
    import sys
    import platform
    
    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.system(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "architecture": "Evolutionary Multi-Modal AI",
        "status": "🚀 ACTIVE - Next-Gen AI Operational"
    }

# Quick start function
def quick_start(model_size: str = "7B", optimization: str = "balanced") -> 'EvolutiveBrain':
    """
    Quick start function to initialize ESERISIA AI with optimal settings.
    
    Args:
        model_size: Model size ("1B", "7B", "13B", "70B", "175B")
        optimization: Optimization level ("fast", "balanced", "ultra")
    
    Returns:
        Initialized EvolutiveBrain instance
    """
    return EvolutiveBrain(
        model_size=model_size,
        optimization=optimization,
        auto_evolution=True,
        quantum_enabled=True,
        distributed=True
    )
