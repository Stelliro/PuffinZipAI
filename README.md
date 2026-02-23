# PuffinZipAI

**AI-driven compression research platform** — Q-learning and Dueling-DQN agents discover, evolve, and benchmark novel compression strategies through evolutionary optimisation.

> **Working Prototype · v0.9.7**

---

## Features

| Area | Highlights |
|------|-----------|
| **AI Agents** | Tabular Q-learning, GPU-accelerated Q-table (CuPy), Dueling DQN with NoisyNet + PER (PyTorch) |
| **Evolutionary Optimiser** | Tournament / roulette / rank selection, crossover, mutation, elitism, "grandpapi" heritage lineage |
| **Compression Primitives** | Simple & Advanced RLE, BWT, MTF, Delta, BPE, random XOR/permutation/block-shuffle discovery transforms |
| **Benchmarking** | Head-to-head vs gzip / bz2 / lzma / zlib / zstd; Gold-standard checkpoints; tiered success criteria |
| **Progressive Difficulty** | 6-tier complexity scaling, ratio + gold-standard gating, dwell-time requirements |
| **Interfaces** | CLI, Tkinter desktop GUI, Flask web dashboard |
| **GPU Support** | Optional CuPy/Numba acceleration; auto-detects hardware |

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url> PuffinZipAI
cd PuffinZipAI
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Web UI (recommended)

```bash
# Windows
scripts\run_webui_windows.bat

# Linux / macOS
scripts/run_webui.sh
```

Open **http://localhost:5001** (Windows) or **http://localhost:5000** (Linux/macOS).

### 3. Or use the Tkinter GUI

```bash
python run_gui.py
```

### 4. Or use the CLI

```bash
python main_cli.py
```

## Optional Dependencies

The core platform runs on **CPU with no GPU or deep-learning libraries**.
For advanced features, install the optional dependencies:

```bash
# GPU acceleration (requires CUDA 12.x toolkit)
pip install cupy-cuda12x

# Neural-network (Dueling DQN) agents
pip install torch

# Zstandard baseline in benchmarks
pip install zstandard
```

## Project Structure

```
PuffinZipAI/
├── main_cli.py                  # CLI entry point
├── run_gui.py                   # Tkinter GUI launcher
├── webui_server.py              # Flask web dashboard
├── requirements.txt
│
├── puffinzip_ai/                # Core package
│   ├── ai_core.py               #   Q-learning agent
│   ├── reward_system.py         #   Multi-signal reward
│   ├── rle_utils.py             #   RLE compress/decompress
│   ├── novel_compression_generator.py  # Invertible primitive pipelines
│   ├── compression_scaffolding.py      # Reference-method training wheels
│   ├── hybrid_compression_engine.py    # Unified compression interface
│   ├── compression_benchmark.py        # A100-ready benchmark suite
│   ├── gold_standard_benchmark.py      # Head-to-head vs baselines
│   ├── checkpoint_manager.py           # Save / load / compare
│   ├── evolution_core/          #   Evolutionary optimiser
│   ├── gpu_core/                #   CuPy/Numba GPU acceleration
│   ├── nn_core/                 #   Dueling DQN (PyTorch)
│   └── utils/                   #   Benchmark evaluator, GitHub fetcher, tuner
│
├── puffinzip_gui/               # Tkinter desktop GUI
├── webui_static/                # Web UI assets (CSS/JS)
├── webui_templates/             # Web UI HTML
├── scripts/                     # Launcher & packaging scripts
├── examples/                    # Demo scripts
├── docs/                        # Guides & changelog
├── tests/                       # Test suite
└── CODEBASE_INDEX.md            # Detailed architecture reference
```

## Documentation

| Document | Description |
|----------|-------------|
| [CODEBASE_INDEX.md](CODEBASE_INDEX.md) | Full architecture & API reference |
| [docs/WEBUI_GUIDE.md](docs/WEBUI_GUIDE.md) | Web UI usage guide |
| [docs/HYBRID_COMPRESSION_GUIDE.md](docs/HYBRID_COMPRESSION_GUIDE.md) | Hybrid engine architecture |
| [docs/changelog.md](docs/changelog.md) | Development changelog |

## How It Works

1. **Initialisation** — A population of AI agents is created, each with its own Q-table (or DQN) and hyperparameters.
2. **Evaluation** — Every agent compresses a benchmark dataset and is scored on compression ratio, round-trip correctness, speed, and diversity.
3. **Benchmarking** — The best agent is compared head-to-head against gzip, bz2, lzma, zlib, and zstd. If the agent beats *all* baselines on *every* test item, a gold-standard checkpoint is saved.
4. **Breeding** — Top agents are selected, crossed over, and mutated. Heritage ("grandpapi") lineage carries proven pipelines forward.
5. **Difficulty Scaling** — Benchmark complexity advances through six tiers, gated by compression ratio and gold-standard win rate.
6. **Repeat** — The loop continues until the generation limit is reached or the user stops training.

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).  
You may use, modify, and share this software freely for any **non-commercial** purpose.  
Commercial use is not permitted without separate written agreement from the licensor.
