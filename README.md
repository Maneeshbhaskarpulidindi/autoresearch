# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## The Idea & How It Works

### Core Concept

The fundamental idea behind autoresearch is a **paradigm shift**: instead of the human writing ML training code, the human writes *instructions for an AI agent* (`program.md`), and the agent writes and iterates on the actual training code (`train.py`). The human becomes a **meta-programmer** — programming the researcher, not the research.

### The Autonomous Experiment Loop

The agent follows a simple but powerful **greedy hill-climbing** loop:

```
LOOP FOREVER:
  1. Propose a hypothesis (e.g. "increase learning rate to 0.04")
  2. Edit train.py with the change
  3. Git commit
  4. Run training (fixed 5-minute budget)
  5. Read the result metric (val_bpb)
  6. If improved → KEEP (advance the branch)
     If worse   → DISCARD (git reset)
  7. Log results to results.tsv
```

Each experiment is **exactly 5 minutes** of wall-clock training. This means:
- **~12 experiments per hour**, ~100 overnight while you sleep
- Results are **directly comparable** regardless of what was changed
- The agent finds the **optimal model for your specific hardware**

### Implementation Architecture

The system has a strict separation of concerns:

| Component | File | Who Controls It |
|-----------|------|-----------------|
| Agent instructions | `program.md` | Human edits this |
| Model + optimizer + training loop | `train.py` | Agent edits this |
| Data, tokenizer, evaluation metric | `prepare.py` | Nobody (read-only) |

The agent can modify **everything** in `train.py`: model architecture (depth, width, attention heads), optimizer hyperparameters (learning rates, momentum, weight decay), batch size, learning rate schedule, activation functions, and more. The evaluation function in `prepare.py` is the immutable ground truth — ensuring fair comparison across all experiments.

### What the Agent Tunes

The default `train.py` exposes these major parameter categories:

- **Architecture**: `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`
- **Optimizer**: `MATRIX_LR`, `EMBEDDING_LR`, `UNEMBEDDING_LR`, `WEIGHT_DECAY`, `ADAM_BETAS`
- **Schedule**: `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`
- **Batch sizing**: `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`
- **And everything else**: activation functions, normalization, RoPE frequency, value embeddings, etc.

The agent uses its knowledge of deep learning to propose changes, and the keep/discard mechanism acts as a natural filter — only improvements survive.

---

## Using Autoresearch With Your Own Custom Model

The autoresearch pattern is **model-agnostic**. You can adapt it to optimize any ML model — image classifiers, object detectors, speech models, RL agents, etc. Here's how:

### Step 1: Create Your Training Script

Create a `train.py` (or equivalent) that:
- Contains your full model definition, optimizer, and training loop
- Has clearly labeled hyperparameters at the top that the agent can tune
- Runs for a **fixed time budget** (e.g. 5 minutes)
- Prints a single evaluation metric at the end

```python
# Example: Custom image classifier train.py
# --- Hyperparameters (agent tunes these) ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_LAYERS = 4
HIDDEN_DIM = 256
DROPOUT = 0.1
OPTIMIZER = "adamw"
AUGMENTATION = "basic"

# ... model definition, training loop ...

# Print result in a parseable format
print(f"val_accuracy: {val_acc:.6f}")
print(f"val_loss: {val_loss:.6f}")
print(f"peak_vram_mb: {peak_mem:.1f}")
```

### Step 2: Create Your Evaluation (Keep It Separate & Read-Only)

Put your evaluation logic in a **separate file** that the agent cannot modify:

```python
# evaluate.py (read-only, like prepare.py)
def evaluate(model, val_loader):
    # Your fixed evaluation metric
    # Returns a single number: lower = better (or higher = better)
    return val_loss
```

### Step 3: Write Your `program.md`

Adapt the instructions to your model. The key sections are:

```markdown
# My Custom Model Research

## What you CAN do:
- Modify train.py — architecture, optimizer, hyperparameters, augmentation, etc.

## What you CANNOT do:
- Modify evaluate.py or data loading
- Install new packages

## The goal: get the lowest val_loss (or highest val_accuracy)

## Experiment loop:
1. Edit train.py with an idea
2. Git commit
3. Run: python train.py > run.log 2>&1
4. Check: grep "^val_loss:" run.log
5. If improved → keep. If worse → git reset.
6. Log to results.tsv
7. Repeat forever.
```

### Step 4: Run the Agent

Point your AI agent (Claude, Codex, Gemini, etc.) at the repo and say:

```
Read program.md and start experimenting. Establish a baseline first.
```

### Example Adaptations

| Use Case | Metric to Optimize | What Agent Tunes |
|----------|-------------------|------------------|
| Image Classification | `val_accuracy` (↑) | LR, architecture, augmentation, optimizer |
| Object Detection (YOLO) | `val_mAP` (↑) | Anchors, backbone, neck, augmentation |
| Text Classification | `val_f1` (↑) | Embedding dim, layers, dropout, LR |
| Speech Recognition | `val_wer` (↓) | Model size, spectrogram params, decoder |
| Reinforcement Learning | `episode_reward` (↑) | Network size, reward shaping, exploration |
| Image Generation | `val_fid` (↓) | UNet depth, noise schedule, LR |

### Tips for Best Results

1. **Keep experiments fast** — 5-10 minutes each is ideal. The agent benefits from rapid iteration.
2. **One clear metric** — the keep/discard decision must be unambiguous.
3. **Label your hyperparameters** — put them at the top of the file with comments so the agent understands what each one does.
4. **Start simple** — let the agent establish a baseline with default settings first.
5. **Don't over-constrain** — the more freedom the agent has, the more creative its solutions can be.
6. **Use git branches** — each experiment run is a commit, so you get a full history of what was tried.

### Key Constraints to Be Aware Of

- **Greedy search**: the agent does hill-climbing, so it can get stuck in local optima. It won't explore radically different architectures if the current one is "good enough."
- **Fast feedback required**: models that take hours to train won't benefit much — the agent needs rapid iteration cycles.
- **Agent knowledge**: the agent proposes ideas based on its training data, so it works best for well-studied model types (transformers, CNNs, etc.).
- **Single metric**: if your task has multiple objectives (accuracy vs. speed vs. size), you need to combine them into one number.

---

## License

MIT
