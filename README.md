## Fork to improve this architecture

# Transformer Decoder for Quantum Error Correction

A graph-based neural decoder for quantum error correction (QEC) on Tanner graphs.
Supports two processor backends — a lightweight **GNN** (GCN-style message passing)
and a **graph transformer** (multi-head attention with edge features and tanner/dual
mode alternation). Both use adaptive depth that scales with code distance, trained
via supervised learning from a MWPM teacher with optional RL fine-tuning.

## Features

- **Two graph processors**: GNN (~123K params) and graph transformer (~234K params)
- **Adaptive depth** proportional to code distance for both processors
- **Repetition and surface code** constructors (no external datasets needed)
- **Curriculum learning** with multi-phase training across code distances
- **Multiple training objectives**: cross-entropy (supervised) and REINFORCE/GRPO (RL)
- **MWPM baseline** via PyMatching for comparison
- **YAML-driven configuration** for reproducible experiments

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.7+.

## Quick Start

```bash
# Full training + evaluation on surface codes (requires GPU)
python main.py all --config config.yaml

# Quick CPU smoke test (~1 min)
python main.py train --config config.smoke.yaml

# Training only
python main.py train --config config.yaml

# Evaluation only (requires a trained checkpoint)
python main.py eval --config config.yaml
```

## Project Structure

```
src/
  base/          # Problem definition
    dem.py         # DetectorErrorModel: Tanner graph + Markov error model
    tanner.py      # Sparse bipartite graph representation
    codes.py       # Repetition and surface code constructors
  core/          # Decoders and simulation
    decoder.py     # Decoder base class
    baseline_decoder.py  # GraphDecoder (GNN or transformer backend)
    mwpm_decoder.py      # MWPM baseline (PyMatching)
    simulator.py   # Error state simulation
    player.py      # Base class for simulator/decoder
    layers/        # Graph processor building blocks
      graph_processors.py  # AdaptiveSharedGNN + AdaptiveSharedGraphTransformer
      graph_attention.py   # Edge-aware multi-head sparse attention
      graph_transformer.py # GraphTransformerBlock (tanner/dual modes)
      components.py        # RMSNorm, SwiGLU
  task/          # Training and evaluation
    losses.py      # Loss functions (CE, REINFORCE, GRPO)
    training.py    # Trainer with curriculum learning
    evaluation.py  # Logical error rate evaluation
  util/
    tokenizer.py   # Binary vector tokenizer
```

## Configuration

See `config.yaml` for a full reference configuration and `config.smoke.yaml`
for a minimal example. Key sections:

- **training**: code type/distance, model architecture, optimizer, curriculum phases
- **evaluation**: distances, error rates, decoders, output directory

### Switching between GNN and Transformer

Set `processor_type` in the training config:

```yaml
# GNN mode (default) — lightweight, GCN-style message passing
processor_type: gnn

# Transformer mode — multi-head graph attention with edge features
processor_type: transformer
n_heads: 4
graph_modes: [tanner, dual]   # alternating tanner/dual blocks
update_edge_attr: true
```

## Results

Both architectures were trained for 300 epochs with curriculum learning (phase 1:
d=3,5,7 for 100 epochs; phase 2: d=3,5,7,9,11 for 200 epochs), batch size 512,
MWPM teacher, and mixed precision on a single GPU.

### Logical Error Rate (LER) — GNN (123K params)

| d  | p=0.01 | p=0.03 | p=0.05 | p=0.08 | p=0.10 |
|----|--------|--------|--------|--------|--------|
| 3  | 0.0018 | 0.0143 | 0.0381 | 0.0836 | 0.1199 |
| 5  | 0.0003 | 0.0062 | 0.0251 | 0.0758 | 0.1247 |
| 7  | 0.0000 | 0.0037 | 0.0192 | 0.0795 | 0.1394 |
| 9  | 0.0000 | 0.0021 | 0.0163 | 0.0805 | 0.1502 |
| 11 | 0.0000 | 0.0017 | 0.0139 | 0.0827 | 0.1644 |

### Logical Error Rate (LER) — Transformer (234K params)

| d  | p=0.01 | p=0.03 | p=0.05 | p=0.08 | p=0.10 |
|----|--------|--------|--------|--------|--------|
| 3  | 0.0019 | 0.0152 | 0.0399 | 0.0869 | 0.1239 |
| 5  | 0.0004 | 0.0078 | 0.0301 | 0.0851 | 0.1376 |
| 7  | 0.0001 | 0.0055 | 0.0257 | 0.0918 | 0.1571 |
| 9  | 0.0001 | 0.0042 | 0.0259 | 0.1029 | 0.1795 |
| 11 | 0.0001 | 0.0041 | 0.0241 | 0.1124 | 0.2000 |

### MWPM Baseline (for reference)

| d  | p=0.01 | p=0.03 | p=0.05 | p=0.08 | p=0.10 |
|----|--------|--------|--------|--------|--------|
| 3  | 0.0017 | 0.0147 | 0.0359 | 0.0824 | 0.1175 |
| 5  | 0.0003 | 0.0062 | 0.0240 | 0.0758 | 0.1241 |
| 7  | 0.0001 | 0.0026 | 0.0160 | 0.0718 | 0.1257 |
| 9  | 0.0000 | 0.0012 | 0.0112 | 0.0647 | 0.1293 |
| 11 | 0.0000 | 0.0004 | 0.0069 | 0.0583 | 0.1303 |

**Key findings:** The GNN matches MWPM at d=3-5 (1.0x ratio) and stays within
1.0-1.3x up to d=11. The transformer, despite having nearly 2x the parameters,
consistently underperforms the GNN. Locality (GCN message passing) outperforms
global attention (graph transformer) for this task.

### Reproducing

```bash
# GNN (recommended)
python main.py all --config config_gnn.yaml

# Transformer
python main.py all --config config_transformer.yaml
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
