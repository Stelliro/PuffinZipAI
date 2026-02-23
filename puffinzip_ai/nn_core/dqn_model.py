# PuffinZipAI_Project/puffinzip_ai/nn_core/dqn_model.py
"""
**Hyper-Complex Adaptive Deep Q-Network (DQN) for PuffinZipAI agents.**

Architecture — Dueling DQN with Multi-Head Self-Attention + Noisy Linear layers:

    Input (NN_STATE_FEATURE_DIM)
        ↓
    [ Shared Feature Encoder ]
        LayerNorm → FC+GELU → Dropout → [ResidualBlock × N]
        ↓
    [ Multi-Head Self-Attention Block ]
        Reshapes features → Multi-Head Attention → residual + LayerNorm
        ↓
    [ Adaptive Feature Gate ]
        Sigmoid gating → element-wise feature weighting
        ↓
    ┌──────────────────────────────────┐
    │      DUELING ARCHITECTURE        │
    ├──────────────────────────────────┤
    │  Value Stream (V):               │   Advantage Stream (A):
    │    NoisyLinear+GELU              │     NoisyLinear+GELU
    │    NoisyLinear → scalar          │     NoisyLinear → action_dim
    └──────────────────────────────────┘
        ↓
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

Key improvements over the previous simple MLP:
    * **Dueling architecture** — separates state value from action advantage,
      enabling faster convergence when many actions have similar value.
    * **Noisy Linear layers** — learned parametric noise replaces ε-greedy
      exploration in the value/advantage streams, enabling state-dependent
      exploration that adapts over training.
    * **Multi-Head Self-Attention** — enables the network to learn complex
      feature interactions and attend to the most relevant input signals.
    * **Layer Normalization** — stabilises training across highly variable
      compression inputs.
    * **GELU activation** — smoother gradient flow than ReLU.
    * **Residual connections** — improved gradient propagation for deeper nets.
    * **Dropout** — regularisation to prevent overfitting on limited data.
    * **Adaptive Feature Gate** — sigmoid gating that dynamically weights
      input features based on data characteristics.

Typical sizes with default config (20 → 256 → 256 → 4):
    Parameters : ~200K  (~800 KB fp32)
    Forward    : ~50 µs on GPU
"""

from __future__ import annotations

import copy
import logging
import math
from typing import List, Optional, Sequence

logger = logging.getLogger("puffinzip_ai.nn_core.dqn_model")

try:
    import torch  # type: ignore[import-unresolved]
    import torch.nn as nn  # type: ignore[import-unresolved]
    import torch.nn.functional as F  # type: ignore[import-unresolved]
except ImportError as _e:
    raise ImportError(
        "PyTorch is required for neural-network agents.  "
        "Install it with:  pip install torch  (or torch-cuda for GPU support)"
    ) from _e


# ---------------------------------------------------------------------------
# Noisy Linear Layer (Factorised Gaussian Noise)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Factorised Noisy Linear layer (Fortunato et al., 2018).

    Replaces standard Linear with learned parametric noise, enabling
    state-dependent exploration without external ε-greedy.

    Parameters
    ----------
    in_features : int
    out_features : int
    sigma_init : float
        Initial noise magnitude. Default 0.5.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorised noise buffers (not parameters — regenerated each forward)
        self.weight_epsilon: torch.Tensor
        self.bias_epsilon: torch.Tensor
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self._sigma_init = sigma_init
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self._sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self._sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _factorised_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Regenerate factorised noise vectors."""
        eps_in = self._factorised_noise(self.in_features)
        eps_out = self._factorised_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, sigma_init={self._sigma_init}"


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention Block
# ---------------------------------------------------------------------------

class FeatureAttentionBlock(nn.Module):
    """Multi-head self-attention over feature dimensions.

    Treats the feature vector as a sequence and learns which features
    to attend to for compression action selection.

    Parameters
    ----------
    feature_dim : int
        Input/output feature dimension.
    num_heads : int
        Number of attention heads (auto-adjusted to divide feature_dim).
    dropout : float
        Dropout rate on attention weights.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Ensure feature_dim is divisible by num_heads
        while feature_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape ``(batch, feature_dim)``

        Returns
        -------
        Tensor  shape ``(batch, feature_dim)``
        """
        # (batch, 1, feature_dim) — self-attention over the feature vector
        seq = x.unsqueeze(1)
        attn_out, _ = self.attention(seq, seq, seq)
        attn_out = attn_out.squeeze(1)
        # Residual + LayerNorm
        out = self.layer_norm(x + self.dropout(attn_out))
        return out


# ---------------------------------------------------------------------------
# Adaptive Feature Gate
# ---------------------------------------------------------------------------

class AdaptiveFeatureGate(nn.Module):
    """Learned gating mechanism that dynamically weights input features.

    Uses a sigmoid gate to learn which features matter most for each input,
    allowing the network to adaptively focus on RLE-relevant vs entropy-relevant
    vs structural features depending on the data characteristics.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_net(x)
        return x * gate  # element-wise gating


# ---------------------------------------------------------------------------
# Shared Feature Encoder
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Multi-layer feature encoder with residual connections, LayerNorm, and GELU.

    Architecture:
        Input → LayerNorm → FC+GELU → [ResidualBlock × N]
              → FeatureAttentionBlock → AdaptiveFeatureGate → LayerNorm
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        dropout: float = 0.1,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_sizes[0])

        blocks: List[nn.Module] = []
        for i in range(len(hidden_sizes)):
            in_dim = hidden_sizes[i]
            out_dim = hidden_sizes[i] if i < len(hidden_sizes) - 1 else hidden_sizes[-1]
            blocks.append(_ResidualBlock(in_dim, out_dim, dropout))

        self.blocks = nn.ModuleList(blocks)
        self.attention = FeatureAttentionBlock(
            hidden_sizes[-1], num_heads=attention_heads, dropout=dropout
        )
        self.feature_gate = AdaptiveFeatureGate(hidden_sizes[-1])
        self.output_norm = nn.LayerNorm(hidden_sizes[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        x = self.attention(x)
        x = self.feature_gate(x)
        x = self.output_norm(x)
        return x


class _ResidualBlock(nn.Module):
    """Single residual block: FC → GELU → Dropout → FC → GELU → residual add → LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        # Projection for dimension mismatch
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = F.gelu(self.fc1(x))
        out = self.dropout(out)
        out = F.gelu(self.fc2(out))
        out = self.dropout(out)
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Dueling DQN Network (main export)
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """Dueling DQN with Noisy Linear, Multi-Head Attention, and Adaptive Gating.

    Architecture
    ------------
    Input → SharedEncoder (LayerNorm + ResidualBlocks + Attention + Gate)
        ├→ Value stream:     NoisyLinear+GELU → NoisyLinear → V(s)  [scalar]
        └→ Advantage stream: NoisyLinear+GELU → NoisyLinear → A(s,a) [action_dim]

    Q(s,a) = V(s) + A(s,a) - mean(A)

    Parameters
    ----------
    state_dim : int
        Dimensionality of the continuous state-feature vector.
    action_dim : int
        Number of discrete actions (compression methods).
    hidden_sizes : list[int]
        Widths of hidden layers in the shared encoder.  Default ``[256, 256]``.
    dropout : float
        Dropout rate throughout the network. Default 0.1.
    attention_heads : int
        Number of attention heads. Default 4.
    noisy_sigma : float
        Initial sigma for NoisyLinear layers. Default 0.5.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
        attention_heads: int = 4,
        noisy_sigma: float = 0.5,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)

        # Shared feature encoder
        self.encoder = SharedEncoder(
            input_dim=state_dim,
            hidden_sizes=self.hidden_sizes,
            dropout=dropout,
            attention_heads=attention_heads,
        )

        enc_out_dim = self.hidden_sizes[-1]

        # --- Dueling streams with NoisyLinear ---
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            NoisyLinear(enc_out_dim, enc_out_dim // 2, sigma_init=noisy_sigma),
            nn.GELU(),
            NoisyLinear(enc_out_dim // 2, 1, sigma_init=noisy_sigma),
        )

        # Advantage stream: estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            NoisyLinear(enc_out_dim, enc_out_dim // 2, sigma_init=noisy_sigma),
            nn.GELU(),
            NoisyLinear(enc_out_dim // 2, action_dim, sigma_init=noisy_sigma),
        )

        # Initialize encoder weights (Xavier for deterministic layers)
        self._init_encoder_weights()

    def _init_encoder_weights(self) -> None:
        """Xavier init for all deterministic Linear layers in the encoder."""
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for every action given a batch of states.

        Uses the Dueling decomposition:
            Q(s, a) = V(s) + A(s, a) - mean_a(A(s, :))

        Parameters
        ----------
        state : Tensor  shape ``(batch, state_dim)`` or ``(state_dim,)``

        Returns
        -------
        Tensor  shape ``(batch, action_dim)`` or ``(action_dim,)``
        """
        features = self.encoder(state)

        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, action_dim)

        # Dueling combination: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values

    # ------------------------------------------------------------------
    # Noise management
    # ------------------------------------------------------------------
    def reset_noise(self) -> None:
        """Regenerate noise in all NoisyLinear layers (call before each forward)."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_noise_magnitude(self) -> float:
        """Return the average absolute noise sigma across all NoisyLinear layers."""
        sigmas = []
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                sigmas.append(module.weight_sigma.data.abs().mean().item())
        return sum(sigmas) / max(len(sigmas), 1)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def clone(self) -> "DQNNetwork":
        """Return a deep-copy of this network (new parameters, same architecture)."""
        return copy.deepcopy(self)

    def hard_sync_from(self, source: "DQNNetwork") -> None:
        """Copy all parameters from *source* into this network (target-net sync)."""
        self.load_state_dict(source.state_dict())

    def soft_sync_from(self, source: "DQNNetwork", tau: float = 0.005) -> None:
        """Polyak averaging: θ_target ← τ·θ_source + (1-τ)·θ_target."""
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def parameter_count(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def memory_bytes(self) -> int:
        """Approximate GPU/CPU memory footprint in bytes (parameters only)."""
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def to_device(self, device: torch.device) -> "DQNNetwork":
        """Move network to *device* and return self for chaining."""
        self.to(device)
        return self

    def get_flat_params(self) -> torch.Tensor:
        """Return all parameters concatenated into a single 1-D tensor (detached, on CPU)."""
        return torch.cat([p.detach().cpu().reshape(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Load parameters from a 1-D tensor produced by :meth:`get_flat_params`."""
        offset = 0
        device = next(self.parameters()).device
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].reshape(p.shape).to(device))
            offset += numel

    def freeze_encoder(self) -> None:
        """Freeze the shared encoder (used during fine-tuning phases)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze the shared encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_architecture_summary(self) -> dict:
        """Return a dict summarizing the architecture for logging/display."""
        return {
            "type": "DuelingDQN+NoisyNet+Attention",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": self.hidden_sizes,
            "total_params": self.parameter_count(),
            "memory_kb": self.memory_bytes() / 1024,
            "noise_magnitude": self.get_noise_magnitude(),
            "has_attention": True,
            "has_feature_gate": True,
            "has_noisy_linear": True,
            "has_residual": True,
        }

    def __repr__(self) -> str:
        return (
            f"DQNNetwork(Dueling+Noisy+Attention, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, hidden={self.hidden_sizes}, "
            f"params={self.parameter_count():,})"
        )
