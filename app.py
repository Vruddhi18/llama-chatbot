import gradio as gr
from ctransformers import AutoModelForCausalLM

# Load LLaMA model (CPU-compatible)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/LLaMA-Pro-8B-GGUF", 
    model_file="llama-pro-8b.Q4_K_M.gguf",
    model_type="llama",
    gpu_layers=0,
    batch_size=1  # Ensures efficient processing
)

# System Prompt for Direct Responses
SYSTEM_PROMPT = """You are an AI assistant. Answer **directly** and **concisely**. Avoid unnecessary explanations or small talk."""

# Chat function with direct responses
def chat(message, history):
    history_str = "\n".join([f"User: {entry[0]}\nAI: {entry[1]}" for entry in history[-5:]])  # Keep last 5 exchanges
    prompt = f"{SYSTEM_PROMPT}\n\n{history_str}\nUser: {message}\nAI:"
    
    response = model(prompt, temperature=0.3, max_new_tokens=50, stop=["\nUser:"])
    return response.strip()

# Gradio UI
iface = gr.ChatInterface(fn=chat, title="LLaMA 3 Chatbot", theme="compact")

# Launch app
iface.launch()