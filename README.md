# 🧠 From 1 Neuron to a Generative AI

> Learning deep learning by reimplementing it **from scratch** — from hand-derived gradient descent in NumPy to a 47M-parameter Transformer trained on a real web-scale corpus.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-from%20scratch-013243?logo=numpy&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="GPU" src="https://img.shields.io/badge/CUDA-tested%20on%20RTX%203060%20Ti-76B900?logo=nvidia&logoColor=white">
</p>

---

## 🎯 The idea

**No layer is used before it's understood.**

This repository is three folders, read in order: [`neuronne/`](neuronne/) derives gradients by hand in NumPy, [`gpt/`](gpt/) rebuilds a Transformer language model from that same foundation in PyTorch, and [`camille0/`](camille0/) scales the result up to a real, large-scale text corpus with a custom visualization tool on top. (A fourth folder, [`archive/`](archive/), holds superseded or exploratory scripts — redundant variants and one broken draft — kept for the record but out of the way of the main narrative.) Every number below was **measured by actually re-running the scripts**, not estimated — including the two things that turned out to be broken (see [Honesty note](#-honesty-note), it's worth reading).

```
      3        →      6,570      →      4,225      →     2,141,952     →    47,125,585
  1 neuron        10-class MLP        bigram            MiniGPT            Camille0
   NumPy             NumPy            PyTorch            PyTorch            PyTorch
(hand-derived      (hand-derived    (nn.Embedding)   (multi-head          (trained on
 gradients)         backprop)                         attention)          SlimPajama-6B)
```

---

## 📊 The progression

| # | Step | File | Architecture | Parameters | Status |
|---|------|------|---------------|-----------:|--------|
| 1 | The single neuron | [`neuronne/1_neuronne.py`](neuronne/1_neuronne.py) | 2 → 1 | 3 | ✅ measured |
| 2 | Generic deep MLP | [`neuronne/infini_couche_chiffre.py`](neuronne/infini_couche_chiffre.py) | 64 → 128 → 64 → 1 | 16,641 | ✅ measured |
| 3 | Multi-class softmax | [`neuronne/softmax+crossenthorpy.py`](neuronne/softmax+crossenthorpy.py) | 64 → 64 → 32 → 10 | 6,570 | ✅ measured (bug found & fixed) |
| 4 | Language, by counting | [`gpt/Etape1.py`](gpt/Etape1.py), [`gpt/Etape2.py`](gpt/Etape2.py) | n-gram tables (V², V³) | — | ✅ measured (toy corpus) |
| 5 | Neural baseline | [`gpt/bigram.py`](gpt/bigram.py) | `Embedding(65, 65)` | 4,225 | ✅ measured — full 1.1M-char corpus |
| 6 | Karpathy reference implementation | [`gpt/pytorch_cpu.py`](gpt/pytorch_cpu.py) | 6 blocks, 6 heads, d=384 | 10,788,929 | ✅ measured — best val loss 1.49 |
| 7 | Independent Transformer rewrite | [`gpt/deuxieme_gpt.py`](gpt/deuxieme_gpt.py) | 6 blocks, 6 heads, d=384 | 10,715,201 | 📎 implemented, not trained here |
| 8 | **MiniGPT** (trainer: [`pytorch_gpu.py`](gpt/pytorch_gpu.py), demo: [`chat_bot.py`](gpt/chat_bot.py)) | 4 blocks, 8 heads, d=256 | **2,141,952** | ✅ measured — held-out val loss 1.09 |
| 9 | **Camille0** ([`camille0/`](camille0/)) | 6 blocks, 6 heads, d=384, GPT-2 BPE | **47,125,585** | ✅ measured — trained on SlimPajama-6B |

All parameter counts were computed by instantiating each real class from the actual files and calling `sum(p.numel() for p in model.parameters())` — not estimated by hand.

---

## 🔬 Act I — The neuron, by hand

### [`neuronne/1_neuronne.py`](neuronne/1_neuronne.py) · 3 parameters

A sigmoid perceptron broken into elementary functions: `Z()` (linear combination), `A()` (activation), `L()` (binary cross-entropy), with gradients **derived analytically**, not autograd:

```python
dW = np.dot(X.T, (A - Y)) / m
db = np.mean(A - Y)
```

> **The key idea** — combining a sigmoid with BCE makes the activation's derivative cancel out. That `A − Y` term reappears at the output of the softmax classifier in Act II, and again at the output of every Transformer in this repo.

**Measured:** loss goes from the initial forward pass down to **0.1699** after 1,000 iterations of full-batch gradient descent (final weights `[1.482, -1.098]`, bias `1.133`).

---

## 🔬 Act II — Layers, no framework

### [`neuronne/infini_couche_chiffre.py`](neuronne/infini_couche_chiffre.py) · 16,641 parameters

A network of **arbitrary depth**, parameterized by a plain list of layer sizes — forward and backward are generic loops over a weight dictionary, not hardcoded `W1`, `W2` matrices.

```python
params = neurol_network(X_train, y_train, dim=(128, 64), a=0.1, epochs=10000)
```

| Aspect | Choice |
|---|---|
| Task | sklearn `digits` (8×8 images), binary detection of the digit **3** |
| Init | He initialization — `× √(2/n_prev)` |
| Backprop | full chain `dZ ← Wᵀ·dZ ⊙ A(1−A)`, hand-written |
| Data | positive/negative rebalancing by subsampling |

**Measured: 98.65% test accuracy** (10,000 epochs, balanced train/test split, `random_state=42`).

### [`neuronne/softmax+crossenthorpy.py`](neuronne/softmax+crossenthorpy.py) · 6,570 parameters

The real 10-class problem, not a binary stand-in — numerically-stable **softmax** (max-subtraction trick) and categorical **cross-entropy** on one-hot targets, sigmoid on hidden layers only.

**Measured — and this is where re-running the code paid off:** the first run scored **7.78%**, *worse than random guessing* on 10 balanced classes. Every single test prediction had collapsed onto class `1`. Root cause: `initialize_weights` drew weights from `np.random.rand` (uniform, **always positive**) instead of `np.random.randn` (normal, zero-centered), with random (not zero) biases on top — every hidden unit started biased in the same direction and the softmax head never recovered. The fix is two lines (`rand`→`randn`, random bias→zero bias); re-run after the fix: **91.94% test accuracy**. That fix is already applied in this repo.

---

## 🔬 Act III — Modeling language, before any neural net

### [`gpt/Etape1.py`](gpt/Etape1.py) / [`gpt/Etape2.py`](gpt/Etape2.py) · n-gram tables

Before any network: what a language model *is*. Bigram and trigram models built by pure counting over a small, fully-controlled toy string (`"hello world"`) — deliberately tiny so every number in the pipeline can be inspected by hand.

- Transition matrices of shape `V×V` and `V×V×V`
- **Laplace smoothing** (`+1`) to kill zero probabilities
- **Perplexity** on both train and test — the standard language-modeling metric
- **Temperature** — `p^(1/T)` renormalized, the same creativity/coherence knob reused in Act V

**Measured** (`Etape2.py`, toy corpus, train/test split 80/20): trigram perplexity **5.95** train / **8.00** test; bigram perplexity **6.03** train / **8.40** test. These numbers describe the counting mechanism on 11 characters, not language modeling at scale — that starts in the next act, on the full 1.1M-character corpus.

---

## 🔬 Act IV — The first neural language model

### [`gpt/bigram.py`](gpt/bigram.py) · 4,225 parameters

A single `nn.Embedding(65, 65)`: the row for the current character *is* the logits for the next one. Minimal on purpose — it sets up all the training infrastructure reused for the rest of the repo (`get_batch`, `estimate_loss()` under `@torch.no_grad()`, `model.train()`/`model.eval()` alternation, autoregressive sampling).

**Dataset:** Tiny Shakespeare — 1,115,393 characters, 65-character vocabulary, 90/10 split.

**Measured (full 100,000-step run):** loss drops from **4.73 → ~2.46** within the first ~10,000 steps, then plateaus — exactly what a 1-character-of-context model should do; it has no capacity left to exploit. Final step (99,000/100,000): **train loss 2.458, val loss 2.488**.

Generated sample, before vs. after training (same code, same prompt position):

```text
before (step 0):   pYCXxfRkRZd wc'wfNfT;OLlTEeC K jxqPToTb?bXAUG:C-SGJO-33SM:C?YI3a
after (step 99k):  Men pand, bemary.
                    Yof 'sour menm sora anghy t-e nomes twe ten.
                    NENobeakes aghercobun ws m k s withoumas Fond the wllo INour id, mersed
```

No real words yet (a bigram literally cannot produce any — it has 1 character of memory), but word *lengths*, apostrophe placement, and capitalized character-name-like tokens ("NEN...") already emerge from pure statistics.

---

## 🔬 Act V — Attention and the Transformer

Two independent Transformer implementations live in [`gpt/`](gpt/), exploring the same idea from different angles:

| File | What it is |
|---|---|
| [`gpt/pytorch_cpu.py`](gpt/pytorch_cpu.py) | The reference implementation studied for this project: Andrej Karpathy's *"Let's build GPT"* lecture script, transcribed close to verbatim (6 blocks, 6 heads, d=384, 10.79M parameters) — kept here for comparison, not presented as original design. |
| [`gpt/deuxieme_gpt.py`](gpt/deuxieme_gpt.py) | The same architecture family (6 blocks, 6 heads, d=384, 10.7M parameters), rewritten independently as a second pass at the same ideas. |

**Measured** (`pytorch_cpu.py`, full 5,000-step run, RTX 3060 Ti, ~24 minutes):

| Step | Train loss | Val loss |
|---:|---:|---:|
| 0 | 4.22 | 4.23 |
| 1,000 | 1.39 | 1.61 |
| 2,000 | 1.19 | 1.50 |
| **3,000** | **1.07** | **1.49** ← best val loss |
| 4,000 | 0.97 | 1.52 |
| 4,999 | 0.86 | 1.57 |

Textbook overfitting curve: validation loss bottoms out around step 3,000 while training loss keeps falling — past that point the model is increasingly just memorizing the training split. It also produced the most coherent sample in this repo (temperature-free, argmax-free sampling, straight from the trained model):

```text
O, general, fall too! you ne'er be so lightly,
On ear, or yet a subtle and so day
Having at toward his death in his own: marry,
Signior Hugh Denry, with an English heart,
```

Real words, plausible archaic syntax, character-name-shaped tokens ("Signior Hugh Denry") — the jump in quality from the bigram model (Act IV) to a 6-layer attention model is the single clearest result in this repository.

And the one rewritten with more deliberate engineering choices, trained and evaluated end-to-end:

### `pytorch_gpu.py` (trainer) + `chat_bot.py` (demo) · **MiniGPT**, 2,141,952 parameters

> Naming quirk worth flagging so you don't get confused reading the code: despite the filenames, [`gpt/pytorch_gpu.py`](gpt/pytorch_gpu.py) is the **training** script (calls `train(...)`, saves `shakespeare_gpt.pth` + `char_mapping.pkl`) and [`gpt/chat_bot.py`](gpt/chat_bot.py) is the **inference/demo** script (loads that checkpoint and generates). The name and the role don't match — this README uses the role, not the filename.

What distinguishes this implementation from the Karpathy-style reference above:

- ⚡ **Vectorized attention** — one `Linear(E, E)` projection each for Q, K, V, then `reshape`/`permute` into parallel heads, instead of a Python loop over independent per-head `Linear` layers. This is closer to how production attention is implemented, and meaningfully faster on GPU.
- 🔗 **Weight tying** — the output projection shares its weights with the token embedding table (`proj.weight = embedding_tok.weight`), a standard trick that cuts parameter count and improves generalization.
- 📐 **GELU** instead of ReLU in the feed-forward block.
- 💾 Checkpointing decouples training from inference (`.pth` + `.pkl` vocabulary).

**Measured** (held-out 10% split, freshly re-sampled — not the exact split from the original training run, but methodologically equivalent): **validation loss 1.088**, **perplexity 2.97**.

Real sample from `python chat_bot.py` (prompt `"MENENIUS:"`, temperature 0.7):

```text
MENENIUS:
Let no my unou l h ouso resesaken't siso l, abea,
```

---

## 🔬 Act VI — Camille0: scaling up

[`camille0/`](camille0/) is the most ambitious piece here: a named project with a visible iteration history and a training pipeline that graduates from a local text file to a real, HuggingFace-hosted web corpus.

| File | What changed |
|---|---|
| [`model.py`](camille0/model.py) | First pass: GPT-2 BPE tokenizer, custom `Camille0` architecture, trained on a local `input.txt`. |
| [`model2.py`](camille0/model2.py) | + mixed precision (`torch.amp`), gradient clipping, train/val loss estimation. |
| [`model2_slim.py`](camille0/model2_slim.py) | + trained on **[SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B)** (real, large-scale web text) via HuggingFace `datasets`, with proper block-grouped tokenization and checkpointing. |
| [`neural_flow.py`](camille0/neural_flow.py) | A custom visualization tool: reloads a trained checkpoint and renders how token embeddings evolve **layer by layer** as an HSV-colored GIF. |

Architecture note: `Camille0`'s attention computes Q/K/V projections **once** and splits the result across heads (via a `ModuleList` of lightweight per-head modules) — a third distinct way of implementing multi-head attention in this repo, different from both the Karpathy per-head-`Linear` loop and the MiniGPT reshape/permute approach. Also unlike MiniGPT, its output head is **not** weight-tied to the embedding — a deliberate point of comparison, not an oversight.

**Measured, from the checkpoint that ships with this repo** (`camille0/checkpoints/camille0_weight.pth`, matched to the exact hyperparameters `neural_flow.py` uses to load it: 6 blocks, 6 heads, d=384, block size 128):

- **47,125,585 parameters** (verified both by instantiating the class and by cross-checking against the 183 MB checkpoint file size at float32).
- Generation quality is **honestly a work in progress**: across several prompts, temperatures, and top-k / repetition-penalty settings, output currently degenerates into short repetition loops:

  ```text
  "The quick brown fox jumps over the lazy" → ...lazy center center center center center...
  "Once upon a time"                        → ...time time time time time time time...
  ```

  This is a common failure mode for a model of this size trained for a limited number of steps on a corpus as large as SlimPajama-6B — it has picked up real vocabulary (`element`, `perspective`, `collection`, `2021` all appear in longer samples) but not yet enough long-range coherence to escape repetitive attractor states, even with `model2_slim.py`'s own repetition-penalty decoding. Next step: resume training from this checkpoint and track held-out loss to find when generation quality actually turns the corner (see [Roadmap](#️-roadmap)).
- [`neural_flow.py`](camille0/neural_flow.py) output — a real, already-generated asset in this repo:

  ![Camille0 embedding flow](camille0/visualisations/camille_flow_flow.gif)

---

## 🩺 Honesty note

Every number in this README was obtained by re-running the actual scripts in this repository right before writing it up — not estimated, not carried over from memory. That process caught two real issues, both disclosed above rather than quietly avoided:

1. **A silent training bug** in `softmax+crossenthorpy.py` (positive-only weight init → complete class collapse → below-random accuracy), now fixed.
2. **A real generation-quality limitation** in the Camille0 checkpoint (repetition collapse under multiple decoding strategies), not yet fixed — listed as a known limitation and a roadmap item instead of being papered over with a cherry-picked sample.

If you spot a discrepancy between a number here and what you get locally, trust your own run — hardware, library versions, and (for anything using `torch.multinomial`) plain sampling randomness will all shift results slightly.

---

## 🚀 Installation

Two separate dependency sets — `neuronne/` only needs NumPy-era tooling, `gpt/` and `camille0/` need PyTorch:

```bash
git clone https://github.com/Camave/deeplearning.git
cd deeplearning

# neuronne/ — NumPy fundamentals
pip install numpy matplotlib scikit-learn tqdm

# gpt/ and camille0/ — PyTorch language models
pip install torch tiktoken transformers datasets pillow imageio tqdm
```

**Run the progression** (scripts use paths relative to their own folder, so `cd` into it first):

```bash
# Act I & II — NumPy fundamentals
cd neuronne
python 1_neuronne.py                 # the neuron and its decision boundary
python infini_couche_chiffre.py      # deep MLP, binary digit detection
python "softmax+crossenthorpy.py"    # 10-class digit classification
cd ..

# Act III & IV — counting models, then a neural baseline
cd gpt
python Etape2.py                     # n-grams, perplexity, temperature
python bigram.py                     # ~100k steps — takes a few minutes even on GPU

# Act V — Transformers (chat_bot.py loads the checkpoint already in this repo, no training needed)
python chat_bot.py                   # generate from the trained MiniGPT
python pytorch_gpu.py                # (re)trains MiniGPT from scratch — GPU strongly recommended
cd ..

# Act VI — Camille0 (checkpoint is NOT in the repo — see note below — so start with training)
cd camille0
python model2_slim.py                # downloads SlimPajama-6B, trains — GPU required, large download
python neural_flow.py                # visualize a trained checkpoint as a GIF
```

> **Note on the Camille0 checkpoint:** `camille0/checkpoints/camille0_weight.pth` is ~183 MB — over GitHub's 100 MB hard limit — so it's excluded via `.gitignore` and **not** present after a fresh clone. Run `model2_slim.py` (or `model2.py`) to produce your own, or ask for the weights directly. `gpt/shakespeare_gpt.pth` (8.3 MB), on the other hand, ships in the repo, so `chat_bot.py` and `deuxieme_gpt.py`-style inference work immediately after cloning.

---

## 🧩 What this repository demonstrates

| Skill | Where |
|---|---|
| Deriving and implementing gradients by hand | Acts I & II |
| Generic backprop over arbitrary depth | `infini_couche_chiffre.py` |
| Numerical stability (softmax, log, He init) — and catching it when it silently breaks | Act II, [Honesty note](#-honesty-note) |
| Imbalanced data handling, train/test methodology | Acts II & III |
| Task-appropriate metrics: accuracy, perplexity | Acts II & III |
| Transformer architecture, implemented three independent ways | Act V, Act VI |
| Tensor-level optimization (vectorized attention, weight tying, AMP, grad clipping) | `pytorch_gpu.py`, `camille0/model2.py` |
| Real large-scale data pipeline (HuggingFace `datasets`, block-grouped tokenization) | `camille0/model2_slim.py` |
| Full lifecycle: train → checkpoint → inference, decoupled | `pytorch_gpu.py` → `chat_bot.py` |
| Honest evaluation, including reporting what doesn't work yet | Camille0 generation quality, above |

---

## 📚 References

- A. Karpathy — [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the tutorial studied for Act V; the closest-to-verbatim transcription is kept at [`gpt/pytorch_cpu.py`](gpt/pytorch_cpu.py) for direct comparison against the independent rewrites.
- Vaswani et al. — *Attention Is All You Need* (2017)
- [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) — corpus for Acts IV–V
- [SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) — corpus for Camille0 (Act VI)

---

## 🗺️ Roadmap

- [ ] Resume Camille0 training and track held-out loss to fix the repetition-collapse behavior described above
- [ ] **BPE tokenizer** for the Act IV/V character-level models — everything there is still character-level
- [ ] Learning-rate **scheduler** (warmup + cosine decay) for Camille0
- [ ] **RoPE** instead of learned absolute position embeddings
- [ ] **KV-cache** for faster autoregressive generation
- [ ] Finish wiring [`tok_slim.py`](camille0/tok_slim.py) (French mC4/C4 corpus prep) into the training pipeline — currently a standalone exploration, not yet connected to `model2_slim.py`

---

<p align="center">
  <sub>Personal learning project</sub>
</p>
