EvolutionaryOptimizer = None
EvolvingAgent = None

try:
    from .individual_agent import EvolvingAgent
except ImportError as e_agent:
    EvolvingAgent = None
except Exception as e_gen_agent:
    EvolvingAgent = None

try:
    from .evolutionary_optimizer import EvolutionaryOptimizer as _EvoOptInternal
    if _EvoOptInternal is not None:
        EvolutionaryOptimizer = _EvoOptInternal
except ImportError as e_opt:
    import traceback; traceback.print_exc()
except Exception as e_gen_opt:
    import traceback; traceback.print_exc()

__all__ = []
if EvolutionaryOptimizer is not None:
    __all__.append("EvolutionaryOptimizer")

if EvolvingAgent is not None:
    __all__.append("EvolvingAgent")

__version__ = "0.2.4"