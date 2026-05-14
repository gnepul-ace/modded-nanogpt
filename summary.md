# modded-nanogpt — Repository Summary

---

## 1. Research Goal

### Problem
The repository frames a concrete, competitive benchmark: **train a language model to ≤ 3.28 cross-entropy loss on the FineWeb validation set using 8 NVIDIA H100 GPUs, as fast as possible.** The 3.28 target matches Andrej Karpathy's GPT-2 (small) replication in llm.c, which originally required ~45 minutes and ~10 B tokens. The speedrun asks: _how many of the techniques claimed to improve LLM training efficiency actually work, under fair, reproducible, competitive conditions?_

### Proposed Method
The codebase is a continuously evolving, community-driven **speed-record holder** rather than a single paper submission. Each accepted PR introduces one or more algorithmic or systems improvements and must beat the prior wall-clock record on the same hardware. The current best (record #80, as of early April 2026) trains to target in **≈ 1.406 minutes** — a 1800× wall-clock improvement over the llm.c baseline and a 25× improvement in token efficiency.

### Key Innovations (cumulative across 80+ records)

| Category | Technique |
|---|---|
| **Optimizer** | Muon (SGD-momentum + Newton-Schulz orthogonalisation → replaced with Polar Express), NorMuon (low-rank variance estimator), Cautious Weight Decay, interleaved Adam/NorMuon steps, backward hooks on Adam for async overlap |
| **Architecture** | Rotary embeddings (RoPE) with YaRN scaling, QK-Norm, ReLU² MLP (fused Triton), zero-init projections (muP-like), value embeddings mixed into attention layers, U-Net / skip connections from embedding to every block and block 3→6, "Backout" (single activation reuse for last 3 attn layers), "Smear" (1-token look-back on embeddings), Paired-Head Attention, Bigram Hash Embedding, Multi-Token Prediction head |
| **Precision / Quantisation** | BF16 activations and weights, FP8 matmul for LM head (asymmetric rescale + logit softcap), FP8 gradient storage in cross-entropy backward |
| **Attention** | Flash Attention 3, long-short sliding-window pattern (Gemma 2-inspired), window size warmup with YaRN, sparse attention gate |
| **Data / Batching** | Align batch starts with EoS, max document length = 2048, accumulate 2 gradient steps for embedding/head before update, batch size schedule (8→16→24 × 2048 × 8), sequence length schedule (896→2048) |
| **Comms / Systems** | Reduce-scatter instead of all-reduce, overlapped gradient communication and compute, sparse reduce-scatter for bigram embedding gradients, async data prefetch, custom Triton kernels for symmetric matmul (XXT/XTX), fused linear-relu² MLP, fused softcapped multi-token prediction cross-entropy (CUDA kernel), tiled coalesced transpose-copy/add kernels |

---

## 2. Tech Stack & Dependencies

### Languages
- **Python 3.12** (training, data pipeline, evals)
- **CUDA C / PTX** (inline `torch.cuda._compile_kernel` for the fused CE forward-backward kernel)
- **Triton** (custom kernels: XXT, XTX, linear-relu², transpose-copy/add)

### Core Libraries (`requirements.txt`)

| Package | Role |
|---|---|
| `torch==2.10` (nightly `cu126` in Docker) | DDP training, `torch.compile`, `_scaled_mm`, `FlexAttention` |
| `triton` | Custom GPU kernel authoring |
| `numpy` | CPU-side data handling, sparse comms index computation |
| `huggingface-hub` + `datasets` | Downloading pre-tokenised FineWeb shards |
| `tiktoken` | GPT-2 tokeniser (used in data prep) |
| `kernels` | `get_kernel` import — provides Flash Attention 3 (`flash_attn_varlen_func`) via the `kernels` PyPI package |
| `setuptools`, `typing-extensions==4.15.0` | Build / compatibility |

### Hardware Requirement
- **8× NVIDIA H100 SXM** (official timing). The code is DDP-aware and scales to fewer GPUs via `--nproc_per_node`.
- CUDA 12.6, cuDNN, NCCL (pinned in Dockerfile: `nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04`).
- Flash Attention 3 requires compute capability `sm_90` (H100); the CE CUDA kernel is also compiled for `compute_capability="90"`.

---

## 3. Repository Structure

```
modded-nanogpt/
├── train_gpt.py            # Main training script (~2300 lines) — model + optimizer + training loop
├── train_gpt_medium.py     # GPT-2 Medium track variant (≤2.92 loss target)
├── triton_kernels.py       # All custom Triton/CUDA kernels
├── run.sh                  # Launch: torchrun --standalone --nproc_per_node=8 train_gpt.py
├── Dockerfile              # Reproducible container (CUDA 12.6, Python 3.12, nightly torch)
├── requirements.txt        # Python dependencies
├── data/
│   ├── cached_fineweb10B.py    # Downloads pre-tokenised 10 B-token FineWeb shards from HF Hub
│   ├── cached_fineweb100B.py   # Same for 100 B-token variant
│   ├── cached_finewebedu10B.py # FineWeb-Edu variant
│   └── fineweb.py              # Full tokenisation pipeline (from raw HF dataset)
├── evals/
│   └── hellaswag.py        # HellaSwag zero-shot evaluation (used post-training)
├── records/
│   ├── track_1_short/      # Per-record subdirs with training logs for the ≤3.28 track
│   ├── track_2_medium/     # Per-record logs for the ≤2.92 (GPT-2 Medium) track
│   └── track_3_optimization/ # Systems-only optimisation track logs
└── img/                    # README figures
```

### Key Roles

- **`train_gpt.py`** is the monolithic entry point: it defines the `GPT` model class, `CausalSelfAttention`, `Yarn` (RoPE + YaRN), `NorMuonAndAdam` (combined optimizer), `TrainingManager` (schedule, comms ordering), `TrainingSchedule`/`TrainingStage` (batch size, sequence length, window size, MTP weight schedules), the distributed data generator, and the full training + validation loop.
- **`triton_kernels.py`** houses all performance-critical CUDA/Triton code: symmetric matmul kernels (`XXT`, `XTX`, `ba_plus_cAA`), fused linear-relu² MLP (`FusedLinearReLUSquareFunction`), fused softcapped cross-entropy forward+backward (`FusedSoftcappedCrossEntropy`), coalesced transpose-copy and transpose-add kernels, and the Polar Express orthogonalisation (`polar_express`).
- **`data/cached_fineweb10B.py`** is the recommended data download script: it fetches pre-tokenised `.bin` shards from `kjj0/fineweb10B-gpt2` on Hugging Face, avoiding the ~1 hour tokenisation step.

### Training Entry Point
```bash
python data/cached_fineweb10B.py 9   # download first 900 M tokens
torchrun --standalone --nproc_per_node=8 train_gpt.py
# or via run.sh
```

### Evaluation Entry Point
- Validation loss is computed in-loop every 250 steps and at the end against 10,485,760 FineWeb val tokens.
- HellaSwag is optionally run post-training (`args.run_evals = True`).
- `evals/hellaswag.py` can also be invoked standalone.

---

## 4. Reproducibility

### Setup Steps
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
python data/cached_fineweb10B.py 9   # ~900 M tokens; pass 103 for full 10 B
./run.sh                              # launches 8-GPU torchrun
```
Or with Docker (recommended for exact timing):
```bash
sudo docker build -t modded-nanogpt .
sudo docker run -it --rm --gpus all -v $(pwd):/modded-nanogpt modded-nanogpt python data/cached_fineweb10B.py 8
sudo docker run -it --rm --gpus all -v $(pwd):/modded-nanogpt modded-nanogpt sh run.sh
```

### What Is Fixed
- Dataset: FineWeb 10 B GPT-2-tokenised shards, downloaded from a fixed HF Hub repo (`kjj0/fineweb10B-gpt2`). Val set is always the same 10 M tokens.
- Model architecture: 11-layer, 6-head, head_dim=128, model_dim=768 GPT variant.
- `torch.compile(dynamic=False, fullgraph=True)` eliminates Python overhead and graph recompilations.
- `torch._inductor.config.coordinate_descent_tuning` is **explicitly banned** (adds 30 min compilation).
- An untimed kernel warmup section reruns a subset of training steps on dummy state before the timed loop begins.

### Gaps & Concerns

| Issue | Impact |
|---|---|
| **No random seed is set** anywhere in `train_gpt.py`. Stochasticity comes from data ordering, parameter init, and dropout-free training, but results may differ across runs. The speedrun rules require p < 0.01 statistical significance over multiple runs. | Medium: individual runs are not deterministic; users must aggregate multiple runs. |
| **H100-only Flash Attention 3 + FP8** — the CE kernel and `_scaled_mm` with `float8_e4m3fn` are compiled for `compute_capability="90"`. Running on A100 or consumer GPUs requires code changes. | High: reduces portability. |
| **Hardcoded Triton configs** — block sizes/stage counts in `XXT`/`XTX` are hand-tuned for H100. No autotuning fallback. | Low-Medium: may underperform on other hardware. |
| **`kernels` package dependency** — Flash Attention 3 is loaded via `from kernels import get_kernel`. This package version is not pinned in `requirements.txt`, and its source/contents are opaque. | Medium: silent version drift could change behaviour. |
| **`DATA_PATH` env var** — data paths can be overridden via environment variable but this is not documented in the README. | Low. |
| **`train_gpt_medium.py`** is a separate copy of the training script rather than a parameterised variant. The two files diverge and there is no shared abstraction. | Low for users; medium for maintainers. |
| **No checkpoint resume** — `save_checkpoint` defaults to `False`; there is no `--resume` flag. Interrupted runs cannot be continued. | Medium for long runs. |
| **Commit messages are all "."** — the entire commit history uses a single-character message, providing no change-level documentation. PR logs and the README record table are the only provenance. | Low for end users. |

---

## 5. Code Quality

### Notable Patterns
- **Monolithic single-file design**: `train_gpt.py` is ~2300 lines but intentionally self-contained. All source code is read at startup and logged verbatim for reproducibility (`with open(sys.argv[0]) as f: code = f.read()`). This is a deliberate choice, not an oversight.
- **Parameter banks**: Model weights are stored in flat parameter banks (`qk_bank`, `vo_bank`, `mlp_bank`) rather than per-layer `nn.Module` instances. This enables batched Muon orthogonalisation across all layers' matrices simultaneously and fine-grained sharding across GPUs.
- **Label-based optimizer dispatch**: Every `nn.Parameter` receives a `.label` attribute (auto-assigned from `named_parameters()`). The `NorMuonAndAdam` optimizer uses a `param_table` dict keyed by label to determine update rule, communication mode, LR/WD multipliers, and reshape config — avoiding the fragile `param_groups` pattern.
- **Explicit async communication ordering**: Rather than using PyTorch DDP hooks, gradient reduces are manually scheduled via `scatter_order` and `work_order` lists, enabling overlap with compute and fine-grained control over reduce-scatter vs all-reduce per parameter.
- **`torch.compile` everywhere**: The model forward, Polar Express, and individual optimizer kernels are all compiled, including with `dynamic=False` constraints for performance.
- **Custom `torch.autograd.Function` wrappers**: FP8 matmul (`mm_t_op`), fused MLP (`FusedLinearReLUSquareFunction`), and the CE kernel (`FusedSoftcappedCrossEntropy`) all use the custom op + register_autograd pattern for correct gradient propagation under `torch.compile`.
- **CPU-side 0-D tensors for schedule state**: Learning rate, weight decay, momentum, etc., are stored as CPU scalars and passed as arguments to compiled kernels to avoid graph recompilations when hyperparameters change.

### Potential Anti-Patterns
- **Giant `forward()` method**: The `GPT.forward` is a long, flat sequence of operations with many inlined `unbind`, `view`, `cat` calls that assemble layer inputs from the parameter banks. It is hard to follow but is justified by `torch.compile(fullgraph=True)` requirements.
- **Magic constants**: Several numeric constants (e.g., Polar Express polynomial coefficients, FP8 scale values `100/448`, `1.6/448`, `0.75/448`, CE kernel parameters A=23, B=5, C=7.5) appear without explanation in the code. Their origins are hinted at by contributor handles in comments but are not formally documented.
- **Dual-script duplication**: `train_gpt.py` and `train_gpt_medium.py` share large amounts of code with no shared base.
- **Inline CUDA C string**: The cross-entropy CUDA kernel is defined as a Python string and compiled at import time. This is clever for portability but makes IDE tooling and debugging difficult.
- **TODO comment in production code**: `# TODO - Confirm.` appears on the `TrainingSchedule` instantiation line alongside a commented-out alternative `cooldown_frac`, suggesting the current hyperparameters may not be fully validated.

### Documentation Clarity
- The README is excellent as a **record log and motivation document**: it lists all 80 records, their contributors, timing, and links to run logs.
- **In-code documentation is sparse**: most functions have no docstrings. Exceptions are the optimizer classes (`NorMuonAndAdam`, `ParamConfig`) and the Triton kernels, which include useful comments.
- Contributor attributions via `@handle` comments serve as informal documentation of which technique each code block implements.
- No API docs, no architecture diagram, no hyperparameter table in the README.

---

## 6. Limitations & Issues

### Stated in README
- Results are hardware-specific: **official timing is on 8× H100 from PrimeIntellect**. Consumer GPUs are not supported without code changes.
- Some techniques (e.g., logit softcapping) are explicitly acknowledged as unlikely to scale to very large models.
- `torch.compile` adds ~7 minutes of JIT compilation latency on first run (untimed).
- A `torch._inductor.config.coordinate_descent_tuning` flag was banned after record 21 because it adds 30+ minutes of compilation.
- The 3.28 target is held at ~0.001–0.002 buffer below the threshold to keep validation straightforward.
- Rules explicitly ban modifying train/val token streams, extra inductor flags, and disallow untimed backward passes.

### Identified Gaps
- **No random seed control**: results vary across runs; users must run multiple seeds to assess statistical significance.
- **Flash Attention 3 and FP8 restrict hardware to H100 (sm_90)**: meaningful barrier to community reproduction on other accelerators.
- **`kernels` package is unpinned**: version drift could silently break Flash Attention 3 behaviour.
- **No checkpoint resumption**: long runs cannot be interrupted and resumed.
- **`train_gpt_medium.py` maintenance burden**: two diverging copies with no shared abstraction.
- **Triton configs are hard-coded for H100**: no autotuning on other GPUs.
- **The speedrun metric (val loss on FineWeb) measures a narrow capability**: it does not directly test downstream task performance. The README acknowledges this and provides a 1.5 B scale-up result as partial evidence of generalisability.
- **No unit tests or CI**: correctness is validated entirely by run logs and statistical val-loss tests.
