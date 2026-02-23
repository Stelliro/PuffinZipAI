# PuffinZipAI_Project/puffinzip_ai/nn_core/__init__.py
"""
Neural Network (DQN) core for PuffinZipAI.

Provides Deep Q-Network agents that replace or augment the tabular Q-table
approach with learned representations.  Each agent owns a lightweight MLP
that maps continuous data-features to action Q-values, trained online via
experience replay and a periodically-synced target network.

Modules
-------
dqn_model      – PyTorch DQN network definition (MLP)
replay_buffer  – Circular experience-replay buffer
nn_agent       – ``PuffinZipAI_NN`` agent class (extends ``PuffinZipAI``)
"""

import logging as _logging

_nn_logger = _logging.getLogger("puffinzip_ai.nn_core")

TORCH_AVAILABLE: bool = False
_torch_import_error: str = ""

try:
    import torch as _torch  # noqa: F401
    TORCH_AVAILABLE = True
    _nn_logger.info(f"PyTorch {_torch.__version__} loaded.  CUDA available: {_torch.cuda.is_available()}")
except ImportError as _e:
    _torch_import_error = str(_e)
    _nn_logger.warning(f"PyTorch not installed — NN agents will be unavailable.  ({_e})")
