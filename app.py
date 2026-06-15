from flask import Flask, request, jsonify
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

app = Flask(__name__)

tokenizer = T5Tokenizer.from_pretrained(
    "trained_model"
)

model = T5ForConditionalGeneration.from_pretrained(
    "trained_model"
)

@app.route("/")
def home():

    return {
        "message": "Grammar Correction API Running"
    }

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

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

    return jsonify({
        "input": text,
        "corrected": corrected
    })

if __name__ == "__main__":
    app.run(debug=True)