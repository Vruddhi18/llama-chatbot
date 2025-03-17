from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

app = FastAPI()

# Define request schema
class ChatRequest(BaseModel):
    user_input: str

# Load model on demand (lazy loading)
def get_model():
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length = 1024  # Reduced for efficiency
    dtype = None
    load_in_4bit = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto"
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# Chat history (limit to 3 messages for speed)
chat_history = []

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input

    # Load model only when needed
    model, tokenizer = get_model()

    # Limit chat history to last 3 exchanges
    history_str = "\n".join([f"User: {entry['User']}\nAI: {entry['AI']}" for entry in chat_history[-3:]])
    context = f"{history_str}\nUser: {user_input}" if chat_history else user_input

    # Define chat prompt
    alpaca_prompt = """### Instruction:
You are a helpful AI assistant.

### Input:
{}

### Response:
""".format(context)

    # Tokenize input
    inputs = tokenizer([alpaca_prompt], return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### Response:")[-1].strip()

    # Append to chat history
    chat_history.append({"User": user_input, "AI": response})

    return {"response": response}
