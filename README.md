# 🩺 MedBrain-0.5B

Custom-trained medical language model for structured healthcare answers:
triage assistance, clinical handoffs, patient education.

**Model weights live on Hugging Face (not in this repo):**
👉 https://huggingface.co/suhailult777/MedBrain-0.5B

- ~0.5B params, PyTorch (converted from JAX/Flax training)
- LoRA Rank 16 fine-tune on `Mohammed-Altaf/medical-instruction-100k`
- 3,500+ total downloads · served on Featherless AI

## Quick start

```bash
pip install -r requirements.txt
python inference.py
```

## Files

- `inference.py` – load model from HF and run a medical prompt
- `rag_example.py` – wrap MedBrain in a minimal RAG loop (ChromaDB) over your own documents
- `requirements.txt` – dependencies

## Limitations

Research artifact only. Never use for clinical decision-making or as a
replacement for a licensed physician. LLMs hallucinate – always consult
a certified doctor for medical advice.