"""
Spacecraft Attitude Control via Hybrid Super-Twisting & Type-1/Type-2 Fuzzy Logic Framework.

Reproduction and extension of:
Sarjoughian et al., "Hardware-validated hybrid super-twisting, fuzzy logic, and deep
reinforcement learning framework for spacecraft attitude control", Scientific Reports (2026).
"""

from .spacecraft_env import SpacecraftEnv
from .sta_controller import STAController, ASTAController, HybridSTAFLCController
from .fuzzy_type1 import Type1FLC
from .fuzzy_type2 import IntervalType2FLC
from .proposed_controller import ProposedAGSIT2FLC
from .metrics import evaluate_simulation, evaluate_disturbance_tolerance, dataframe_to_markdown

__all__ = [
    "SpacecraftEnv",
    "STAController",
    "ASTAController",
    "HybridSTAFLCController",
    "Type1FLC",
    "IntervalType2FLC",
    "ProposedAGSIT2FLC",
    "evaluate_simulation",
    "evaluate_disturbance_tolerance",
    "dataframe_to_markdown",
]
