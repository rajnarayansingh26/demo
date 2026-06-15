from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os

app = Flask(__name__)

MODEL_NAME = "rajnarayansingh26/grammar-corrector"

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded successfully")

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Grammar Correction API Running"
    })

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        text = data.get("text", "")

        input_text = f"fix grammar: {text}"

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

        corrected = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return jsonify({
            "input": text,
            "corrected": corrected
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
