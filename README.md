# 🚀 LLaMA 3 CPU Chatbot

This chatbot is powered by the **LLaMA 3** model and runs entirely on **CPU** using `ctransformers`.  
It is designed to be **lightweight, efficient, and responsive** while providing an interactive **Gradio UI** for seamless conversations.

---

## 🌟 Features

✅ **Runs on CPU** – No GPU required, making it accessible on standard hardware  
✅ **Optimized with `ctransformers`** – Faster inference on CPUs  
✅ **Concise & direct responses** – Avoids unnecessary small talk  
✅ **Interactive Gradio UI** – Easy-to-use web interface  
✅ **Maintains chat history** – Context-aware responses  

---

## 🛠️ Installation & Setup

To run this chatbot locally, follow these steps:

### **1️⃣ Install Dependencies**
Ensure you have Python 3.8+ installed, then run:

```bash
pip install gradio ctransformers
```

### **2️⃣ Download the Model**
You need the LLaMA 3 GGUF model. Download it from TheBloke's Hugging Face repository.
Move the .gguf model file to your project directory.

### **3️⃣ Run the Chatbot**

```bash
python app.py
```

## 🤖 Model & Performance
Model Used: LLaMA 3 (8B) - Quantized (Q4_K_M)

Why CPU?: This chatbot is optimized to run without a GPU, making it accessible to more users.

Optimization: Adjusted temperature, response length, and stop tokens for more accurate answers.

## 📌 Example Conversations
![image](https://github.com/user-attachments/assets/dcce192f-8111-4bb1-bdd9-7fb1d457cd32)
