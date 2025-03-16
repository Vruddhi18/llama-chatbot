from unsloth import FastLanguageModel
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Model
max_seq_length = 2048
dtype = None
load_in_4bit = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)
FastLanguageModel.for_inference(model)

chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}
    
    ### Input:
    {}
    
    ### Response:
    {}"""
    
    context = "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in chat_history[-5:]])
    prompt = alpaca_prompt.format("You are a helpful assistant.", context + "\n" + user_input, "")
    
    inputs = tokenizer([prompt], return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
    response = tokenizer.batch_decode(outputs)[0]
    
    chat_history.append({"user": user_input, "ai": response})
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
