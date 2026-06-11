# Fine-Tuning for a Math RAG System

There are two distinct models you can fine-tune in a RAG pipeline, and they solve different problems. You can do one or both.

---

## Part 1: Fine-Tuning the Embedding Model

### What It Does
The embedding model converts text into vectors so that semantically similar content lands close together in vector space. Fine-tuning it teaches the model what "similar" means **in your domain specifically**.

Out of the box, a general embedding model might score `"solve for x in x^2 - 4 = 0"` as dissimilar to `"find the roots of a quadratic equation"` — because it wasn't trained to understand that these are the same thing mathematically. A fine-tuned embedding model learns those equivalences from your data.

### Effect on the System
- Retrieved chunks become more relevant to math queries
- Fewer hallucinations downstream (better context = better answers)
- Handles LaTeX, math notation, and domain vocabulary correctly
- Smaller, cheaper base models can match larger general ones after fine-tuning

### The Process

**1. Collect training pairs**
You need `(query, positive_chunk, [negative_chunk])` triplets. For math:
- Positive pair: `("what is the derivative of sin(x)", "The derivative of sin(x) is cos(x)...")`
- Negative pair: same query paired with an irrelevant chunk

Sources for math training data:
- [MATH dataset](https://github.com/hendrycks/math) — 12,500 competition math problems with solutions
- [Khan Academy](https://www.khanacademy.org/) content (scraped or API)
- OpenStax free textbooks
- Your own family's Q&A history over time

**2. Choose a base embedding model**
Good free starting points:
- `BAAI/bge-small-en-v1.5` — fast, 384-dim, runs on CPU
- `BAAI/bge-base-en-v1.5` — better quality, still manageable
- `sentence-transformers/all-MiniLM-L6-v2` — very lightweight

**3. Fine-tune with contrastive learning**
The standard loss function is **Multiple Negatives Ranking Loss** (via the `sentence-transformers` library). It pushes query vectors toward their positive chunks and away from negatives.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

train_examples = [
    InputExample(texts=["solve x^2 - 4 = 0", "The roots of x^2 - 4 = 0 are x=2 and x=-2..."]),
    # ... more pairs
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
model.save("math-embedding-model")
```

**4. Evaluate**
Use a held-out set of queries and measure retrieval precision (did the right chunks come back?).

### Cost & Hardware

| Hardware | Feasibility | Training Time (1k pairs) | Notes |
|---|---|---|---|
| NVIDIA GPU (RTX 3090/4090) | Excellent | ~10–30 min | Ideal. Full control. |
| Apple Silicon (M1/M2/M3/M4) | Good | ~1–3 hrs | Use `mps` device in PyTorch |
| Google Colab (free T4) | Good | ~30–60 min | Session limits; save checkpoints |
| CPU only | Slow but possible | ~6–24 hrs | Use `bge-small` only |

**Data cost:** Free (open datasets)
**Compute cost:** Free (local or Colab)
**Difficulty:** Medium — well-documented tooling, no exotic setup needed

---

## Part 2: Fine-Tuning the Generation LLM

### What It Does
The generation model reads retrieved context and produces the final answer. Fine-tuning it teaches it **how to reason and explain math** in the style and depth you want — step-by-step, LaTeX-formatted, family-friendly, or Socratic.

Without fine-tuning, a small open-source LLM (e.g., Mistral 7B) can answer math questions but may skip steps, format poorly, or give terse answers. Fine-tuning shapes the *behavior* of the answers.

### Effect on the System
- Answers follow a consistent pedagogical style (e.g., always show work step by step)
- Better LaTeX formatting in outputs
- Handles your family's level (e.g., middle school vs. calculus) more naturally
- Reduces the gap between a small 7B model and GPT-4/Claude on math specifically

### The Process

**1. Choose a base generation model**
Free models strong at math:
- `mistralai/Mistral-7B-Instruct-v0.3` — best general-purpose small model
- `microsoft/Phi-3-mini-4k-instruct` — surprisingly strong at math for its size (3.8B)
- `deepseek-ai/deepseek-math-7b-instruct` — specifically trained on math, strong baseline
- `meta-llama/Meta-Llama-3.1-8B-Instruct` — strong reasoning, requires HuggingFace access

**2. Collect instruction-tuning data**
You need `(instruction, input, output)` triplets in the style you want:
```json
{
  "instruction": "Solve the following math problem step by step.",
  "input": "Find all values of x such that x^2 - 5x + 6 = 0",
  "output": "We need to factor x^2 - 5x + 6.\n\nLook for two numbers that multiply to 6 and add to -5: those are -2 and -3.\n\nSo: (x - 2)(x - 3) = 0\n\nTherefore x = 2 or x = 3."
}
```

Free datasets:
- [MATH dataset](https://github.com/hendrycks/math) — competition problems + solutions
- [GSM8K](https://github.com/openai/grade-school-math) — grade school word problems with chain-of-thought
- [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) — 260k math instruction pairs, ready to use

**3. Fine-tune with QLoRA (the practical approach)**
Full fine-tuning of a 7B model requires ~80GB of VRAM — not feasible locally. **QLoRA** (Quantized Low-Rank Adaptation) is the standard workaround: it quantizes the base model to 4-bit and trains only small adapter layers (~1–2% of parameters). You get 90% of the benefit with 10% of the compute.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load in 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", quantization_config=bnb_config)

# Attach LoRA adapters
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Train
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()
```

**4. Merge and serve**
After training, merge the LoRA adapters back into the base model and run it locally via Ollama or llama.cpp.

### Cost & Hardware

| Hardware | Feasibility | Training Time (10k examples, 3 epochs) | Notes |
|---|---|---|---|
| NVIDIA GPU (RTX 4090, 24GB) | Good | ~4–8 hrs | QLoRA fits in 24GB comfortably |
| NVIDIA GPU (RTX 3090, 24GB) | Good | ~6–10 hrs | Same as 4090, slightly slower |
| NVIDIA GPU (<16GB VRAM) | Marginal | Very slow or OOM | Use Phi-3-mini (3.8B) instead |
| Apple Silicon (M1/M2/M3/M4) | Possible | ~12–24 hrs | Use MLX-LM framework instead of HuggingFace |
| Google Colab (free T4, 16GB) | Possible but painful | ~10–20 hrs | Session kills mid-run; use Colab Pro or Kaggle |
| CPU only | Not recommended | Days | Not practical for 7B models |

**Data cost:** Free (open datasets)
**Compute cost:** Free (local) or ~$5–20 on a cloud GPU rental (Lambda Labs, Vast.ai) for a full run
**Difficulty:** High — more moving parts (quantization, LoRA, data formatting, VRAM management)

---

## Part 3: Fine-Tuning Both

### Effect on the System
This is the full picture: your embedding model retrieves the most mathematically relevant chunks, and your generation model produces well-structured, pedagogically sound answers from them. The two models reinforce each other.

The order that makes sense:
1. Fine-tune the embedding model first (faster, easier, immediate retrieval improvement)
2. Build the RAG pipeline around it
3. Evaluate answer quality with a base generation model
4. Fine-tune the generation model once you have a sense of where it falls short

### Combined Cost Estimate

| Phase | Time | Difficulty |
|---|---|---|
| Embedding fine-tune | 1–3 hrs (any GPU) | Medium |
| Build RAG pipeline | 1–2 days | Medium |
| Generation fine-tune (QLoRA) | 6–24 hrs (GPU needed) | High |
| Total to working system | ~1–2 weeks part-time | — |

---

## Recommended Path for Your Setup

Given the goal (free, math-focused, family use), here is the practical order:

1. **Start with a pre-trained math LLM** (e.g., `deepseek-math-7b-instruct` via Ollama) — no fine-tuning yet, just get the pipeline working
2. **Fine-tune the embedding model** on math Q&A pairs — highest ROI for lowest effort
3. **Evaluate** where answers fall short (style, depth, formatting)
4. **Fine-tune the generation model** with QLoRA using GSM8K + MATH dataset formatted to your preferred answer style

This approach lets you ship something working quickly and fine-tune incrementally rather than spending weeks training before you have anything to test.
