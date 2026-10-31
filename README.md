# MedBrain-0.5B

A small (about 0.5B parameter) medical language model that gives structured answers for **triage assistance, clinical handoff notes and patient education**. Fine-tuned in JAX/Flax, exported to PyTorch, and published on Hugging Face.

**Model:** [huggingface.co/suhailult777/MedBrain-0.5B](https://huggingface.co/suhailult777/MedBrain-0.5B) · 5,800+ downloads

This repo holds the usage code. The weights live on Hugging Face.

## Model details

| | |
|---|---|
| Size | ~0.5B parameters, Qwen2-architecture causal LM (`Qwen2ForCausalLM`) |
| Fine-tuning | LoRA, rank 16 / alpha 16, merged into the base weights |
| Training | custom JAX/Flax training loop with Optax schedulers, then converted to PyTorch `safetensors` |
| Data | [`Mohammed-Altaf/medical-instruction-100k`](https://huggingface.co/datasets/Mohammed-Altaf/medical-instruction-100k) |
| Prompt format | ChatML (`<|im_start|>user … <|im_end|>`) |
| License | Apache-2.0 (as on the Hugging Face card) |

## Quick start

```bash
pip install -r requirements.txt   # torch, transformers, chromadb
python inference.py
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "suhailult777/MedBrain-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

prompt = "A patient presents with sudden shortness of breath and left-sided chest pain. What are the immediate triage steps?"
text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
out = model.generate(**tokenizer(text, return_tensors="pt"), max_new_tokens=150)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## RAG example

`rag_example.py` shows the model grounded in your own notes:
1. Load a few documents into an in-memory **ChromaDB** collection.
2. Retrieve the top matches for a question.
3. Pass them to MedBrain as context in the prompt.

Swap in your own documents to build a simple domain Q&A bot.

## Files

| File | Purpose |
|---|---|
| `inference.py` | Load the model from Hugging Face and answer one medical prompt |
| `rag_example.py` | Minimal retrieve-then-generate loop with ChromaDB |
| `requirements.txt` | `torch`, `transformers`, `chromadb` |

## Limitations

This is a research artifact. It is **not** a medical device and must not be used for clinical decisions. Small models hallucinate, so always consult a licensed clinician.
