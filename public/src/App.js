from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

app = FastAPI()

# Model Settings
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Check Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

FastLanguageModel.for_inference(model)

chat_history = []  # Stores conversation history

# Request model
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    global chat_history
    user_input = request.message

    # Maintain chat history
    history_str = "\n".join([f"User: {entry['User']}\nAI: {entry['AI']}" for entry in chat_history])
    context = f"{history_str}\nUser: {user_input}" if chat_history else user_input

    prompt = f"""### Instruction:
    You are a helpful AI assistant.
    
    ### Input:
    {context}
    
    ### Response:"""

    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### Response:")[-1].strip()

    chat_history.append({"User": user_input, "AI": response})

    return {"response": response}
