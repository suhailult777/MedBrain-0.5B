"""Run MedBrain-0.5B from Hugging Face on a medical prompt."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "suhailult777/MedBrain-0.5B"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    prompt = (
        "A patient presents with sudden shortness of breath and "
        "left-sided chest pain. What are the immediate triage steps?"
    )
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
