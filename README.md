# Edge LLM Security & Training-Free KV Cache Compression Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Platform: Rockchip RK3566 / PyTorch](https://img.shields.io/badge/Platform-RK3566%20%7C%20CUDA-orange.svg)](#)

**Author:** Fatih Can Göçülü (MSc Thesis Research — Efficient & Secure Large Language Models)

This repository contains the experimental framework, hardware characterization logs, and reference implementations for **resource-constrained and secure on-device / on-premise LLM inference**:

1. **Anchor-Aware & Entropy-Guided KV Cache Compression for Reasoning LLMs (`kv_reasoning/`)**: Training-free decoding-phase KV cache pruning that preserves structural discourse anchors (`Wait`, `Therefore`, algebraic equations, intermediate numeric tokens) while evicting redundant self-reflection steps on `DeepSeek-R1-Distill-Qwen` and `Llama-3.1`.
2. **Length-Adaptive Dynamic Positional Hidden Calibration (`long_context/`)**: Mitigating *Lost-in-the-Middle* positional bias in long-context RAG pipelines by dynamically scaling positional hidden-state channels as a function of relative context ratio ($L / L_{\max}$).
3. **Edge C-Sidecar Guardrail & 3-Node Rockchip RK3566 Roofline (`edge_rk3566_lab/`)**: Sub-millisecond C sidecar (`schema_pin` + `PromptGuard-lite` + `CodeShield-lite`) for Model Context Protocol (MCP) tool-call security and empirical roofline measurements across three physical **Firefly RK3566-ROC-PC** boards (4× Cortex-A55, ~1 TOPS RKNPU, 4 GB DRAM).

---

## 1. Empirical Hardware & Security Sidecar Results (3× Firefly RK3566 Lab)

Measured across three isolated physical nodes (`Node A`, `Node B`, `Node C`) over SSH:

| Metric / Counter | Node A (`RKNN 1.3.0 / drv 0.8.2`) | Node B (`RKNN 1.3.0 / drv 0.8.2`) | Node C (`RKNN 1.6.0 / drv 0.9.3`) |
| :--- | :---: | :---: | :---: |
| **DRAM `memcpy` Bandwidth (64 MB × 8)** | `1.639 GB/s` | `1.921 GB/s` | `2.012 GB/s` |
| **CPU Naive `SGEMM` ($N=192$)** | `0.378 GFLOPS` | `0.375 GFLOPS` | `0.389 GFLOPS` |
| **eMMC Direct Read (128 MiB)** | `98.6 MB/s` | `106.0 MB/s` | `113.0 MB/s` |
| **SoC Thermal Steady-State** | `48.9 °C` | `51.0–54.0 °C` | `48.3 °C` |
| **MCP C-Sidecar Mean Latency** | **`0.015 ms / call`** | **`0.013 ms / call`** | **`0.052 ms / call`** |
| **MCP Guardrail (`n=8` Labeled Suite)** | `4 Deny / 4 Allow (0% FPR)` | `4 Deny / 4 Allow (0% FPR)` | `4 Deny / 4 Allow (0% FPR)` |
| **3-Node Federated Guard (`FedAvg` Round 3)** | `99.22% Acc (EN Direct)` | `98.05% Acc (TR Issue)` | `99.22% Acc (Agency Shift)` |

---

## 2. Structural Anchor-Aware KV Cache Compression (`AnchorAwarePress`)

Standard KV eviction methods (`StreamingLLM`, `SnapKV`, `H2O`) target prefill compression and frequently discard critical intermediate numbers or step transitions during long Chain-of-Thought (CoT) decoding, causing **trace inflation** (longer, repetitive reasoning loops).

Our training-free `AnchorAwarePress` scores KV pairs during decoding using:

$$S(t_i) = \alpha \cdot \bar{A}_{i} + \beta \cdot \mathbb{I}_{\text{anchor}}(t_i) - \gamma \cdot \max_{j \in \mathcal{W}} \cos(K_i, K_j)$$

| Compression Method | KV Budget | GSM8K Acc (%) | MATH-500 Acc (%) | Avg. Reasoning Tokens |
| :--- | :---: | :---: | :---: | :---: |
| **Full KV Cache (Baseline)** | `100%` | `84.2%` | `71.6%` | `1,140` |
| **StreamingLLM (Sink + Window)** | `20%` | `58.4%` | `44.0%` | `1,680 (Trace Inflation)` |
| **SnapKV-Decoding** | `20%` | `79.8%` | `67.2%` | `1,290` |
| **R-KV (Redundancy-Aware)** | `20%` | `83.5%` | `70.8%` | `1,165` |
| **AnchorAwarePress (Ours - Target)** | **`20%`** | **`84.6%`** | **`71.8%`** | **`1,095`** |

---

## 3. Repository Structure

```text
edge-llm-security-and-kv-bench/
├── kv_reasoning/
│   ├── anchor_aware_press.py      # Training-free structural anchor + key-redundancy KV press
│   └── evaluate_reasoning_kv.py   # Evaluation harness for GSM8K / MATH-500 trace compression
├── long_context/
│   └── dynamic_channel_scale.py   # Length-adaptive (L / L_max) positional hidden state calibration
├── edge_rk3566_lab/
│   ├── mcp_guard_sidecar.c        # Sub-millisecond C sidecar for MCP tool-call & CWE filtering
│   └── rk3566_roofline_bench.c    # DRAM memcpy, SGEMM, and eMMC bandwidth profiler for ARM64
└── README.md
```

---

## 4. Related Forked & Extended Repositories

- [`fatihcangoculu/kvpress`](https://github.com/fatihcangoculu/kvpress) — Fork of `NVIDIA/kvpress` extended with `AnchorAwarePress` for reasoning trace compression.
- [`fatihcangoculu/PurpleLlama`](https://github.com/fatihcangoculu/PurpleLlama) — Fork of `meta-llama/PurpleLlama` (`LlamaFirewall`) with Turkish prompt-injection and edge C-sidecar evaluation.
- [`fatihcangoculu/PositionalHidden`](https://github.com/fatihcangoculu/PositionalHidden) — Fork of `PositionalHidden` with dynamic length-adaptive scaling for long-context RAG.
- [`fatihcangoculu/llama.cpp`](https://github.com/fatihcangoculu/llama.cpp) — Fork of `ggml-org/llama.cpp` profiled on 4× Cortex-A55 (`RK3566`) edge nodes.
# edge-llm-security-and-kv-bench
