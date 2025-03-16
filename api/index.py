from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch

app = Flask(__name__)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=1024,  # Reduce max sequence length to avoid large outputs
    dtype=None,
    load_in_4bit=True,
    device_map="auto"
)
FastLanguageModel.for_inference(model)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"error": "Empty message received"}), 400

        # Limit input length
        if len(user_input) > 500:
            return jsonify({"error": "Input too long. Limit to 500 characters."}), 400

        # Process input
        inputs = tokenizer([user_input], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)  # Limit output size
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
