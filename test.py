from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

tokenizer = T5Tokenizer.from_pretrained(
    "trained_model"
)

model = T5ForConditionalGeneration.from_pretrained(
    "trained_model"
)

while True:

    text = input("\nEnter Sentence : ")

    if text.lower() == "exit":
        break

    input_text = f"fix grammar: {text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_length=128
    )

    corrected = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("Corrected :", corrected)