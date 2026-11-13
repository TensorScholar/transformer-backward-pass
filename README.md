# The Transformer Backward Pass: Companion Verification Suite
*Derivations, Numerics, and Memory Contracts from Autograd to Tiled Attention*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** [Muhammad Atashi](https://muhammad-atashi.pages.dev)

This repository contains the official open-source companion code and reference verification harnesses for Chapters 1–17 of:
**"The Transformer Backward Pass: Derivations, Numerics, and Memory Contracts from Autograd to Tiled Attention"**.

Every analytical derivation in the text is paired with a standalone, executable reference implementation tested against PyTorch autograd, central finite differences, and explicit mutation controls.

---

## Architecture of the Repository

- **`code/tgd/` — The Shared Core Library:**
  A minimal, production-grade reference implementation of the complete Transformer decoder block (forward pass, reverse mode, and memory-aware operators):
  - Attention, Grouped-Query Attention (GQA), and KV-cache management
  - Online Softmax & Tiling memory contracts
  - RoPE (Rotary Position Embedding) and frequency scaling
  - RMSNorm, SwiGLU, and fused residuals
  - LoRA low-rank parameter-efficient fine-tuning
  - Mixture-of-Experts (MoE) token dispatch & auxiliary balancing
  - Quantization (FP8, INT8/INT4 scales and outlier isolation)
  - AdamW optimization dynamics

- **`code/chNN/` — Chapter Verification Harnesses (Chapters 1–17):**
  Standalone scripts verifying each chapter's exact mathematical and numerical claims:

| Chapter | Topic | Key Verification Script | Primary Focus |
|---|---|---|---|
| **Ch 01** | Introduction & Canonical Forward | `code/ch01/ch01_canonical_forward.py` | Reference decoder architecture, notation contract, four-account memory & FLOP ledger |
| **Ch 02** | The Matrix-VJP Bridge | `code/ch02/ch02_vjp_bridge.py` | Matrix inner products, Frobenius adjoints, 4-step transpose recipe |
| **Ch 03** | From Scalars to Tensors | `code/ch03/ch03_tensor_chain.py` | Tensor contraction adjoints, index-free chain rule |
| **Ch 04** | The Canonical Graph | `code/ch04/ch04_graph_ledger.py` | Graph ledger, parameter registry, shape consistency |
| **Ch 05** | Residual Contract | `code/ch05/ch05_residual_contract.py` | Residual stream identity, accumulation invariant |
| **Ch 06** | The Reverse Pass | `code/ch06/ch06_verify.py`<br>`code/ch06/ch06_two_head_verify.py` | Multi-head masked-attention VJP, manual vs. autograd agreement |
| **Ch 07** | Training Dynamics | `code/ch07/ch07_training_harness.py`<br>`code/ch07/ch07_attention_scaling.py`<br>`code/ch07/ch07_experiment_checks.py` | Scaling laws, gradient clipping, norm placement sensitivity, CSV table checks |
| **Ch 08** | The Optimizer | `code/ch08/ch08_optimizer_verify.py` | AdamW moment history, scale invariance, update-norm damping |
| **Ch 09** | Scaled Generation | `code/ch09/ch09_generation_verify.py` | Incremental KV-cache decode, prefill vs. decode equivalence |
| **Ch 10** | Precision & Numerics | `code/ch10/ch10_numerics_verify.py` | IEEE-754 precision, cancellation bounds, FP16/BF16/FP64 margins |
| **Ch 11** | Memory-Access Contracts & Tiling | `code/ch11/ch11_kernels_verify.py` | SRAM vs. HBM traffic simulation, online softmax tiling, GQA backprop |
| **Ch 12** | Grouped-Query Attention (GQA) | `code/ch12/ch12_gqa_verify.py` | KV broadcast backward, group-sum merge proof, memory savings |
| **Ch 13** | RoPE & Context Scaling | `code/ch13/ch13_rope_verify.py` | 2D block-diagonal rotation, NTK-aware & linear scaling adjoints |
| **Ch 14** | Quantization | `code/ch14/ch14_quant_verify.py` | Absmax vs. block-wise scaling, outlier noise isolation |
| **Ch 15** | PEFT / LoRA | `code/ch15/ch15_lora_verify.py` | Low-rank forward/backward, weight merge, memory footprint |
| **Ch 16** | Mixture-of-Experts (MoE) | `code/ch16/ch16_moe_verify.py` | Top-$k$ routing, auxiliary loss gradients, expert capacity balance |
| **Ch 17** | Final Synthesis | `code/ch17/ch17_model_verify.py` | Full 91M decoder integration, GQA + RoPE + SwiGLU end-to-end |

- **`figures/` — Vector Diagrams & Graphs:**
  Committed SVGs and Graphviz `.dot` source graphs documenting the reverse-pass computational flow and experimental findings.

---

## Quickstart

### 1. Setup Environment

```bash
git clone https://github.com/TensorScholar/transformer-backward-pass.git
cd transformer-backward-pass

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Verification Harnesses

Run any chapter verification directly with Python:

```bash
# Verify Chapter 1 canonical decoder forward baseline & memory ledger:
python3 code/ch01/ch01_canonical_forward.py

# Verify Chapter 2 Matrix-VJP bridge:
python3 code/ch02/ch02_vjp_bridge.py

# Verify Chapter 6 full backward pass & multi-head VJPs:
python3 code/ch06/ch06_verify.py
python3 code/ch06/ch06_two_head_verify.py

# Verify Chapter 11 memory-hierarchy simulation & tiled GQA kernels:
python3 code/ch11/ch11_kernels_verify.py

# Verify Chapter 17 full decoder model synthesis:
python3 code/ch17/ch17_model_verify.py
```

### 3. Regenerating Publication Figures

```bash
python3 code/tools/regenerate_figures.py
python3 code/tools/render_ch11_figures.py
```

---

## License

This companion verification codebase is released under the [MIT License](LICENSE).
