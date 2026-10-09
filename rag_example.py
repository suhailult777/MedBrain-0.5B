"""Minimal RAG loop: retrieve domain notes with ChromaDB, answer with MedBrain."""
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "suhailult777/MedBrain-0.5B"


def main() -> None:
    client = chromadb.Client()
    docs = client.create_collection("medical_notes")
    docs.add(
        documents=[
            "Aspirin is commonly used for pain relief and inflammation.",
            "Triage chest pain by checking vitals, ECG, and cardiac enzymes.",
        ],
        ids=["doc1", "doc2"],
    )
    query = "How should chest pain be triaged?"
    hits = docs.query(query_texts=[query], n_results=2)
    context = "\n".join(hits["documents"][0])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    prompt = (
        f"<|im_start|>user\nContext:\n{context}\n\n"
        f"Question: {query}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
