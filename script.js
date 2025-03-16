async function sendMessage() {
    let userInput = document.getElementById("userInput").value.trim();
    if (!userInput) return;

    let chatbox = document.getElementById("chatbox");

    // Append user message
    let userMessage = document.createElement("div");
    userMessage.className = "message user";
    userMessage.textContent = `You: ${userInput}`;
    chatbox.appendChild(userMessage);
    document.getElementById("userInput").value = "";
    chatbox.scrollTop = chatbox.scrollHeight;  // Auto-scroll to the latest message

    try {
        let response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userInput })
        });

        let data = await response.json();

        // Append AI response
        let aiMessage = document.createElement("div");
        aiMessage.className = "message ai";
        aiMessage.textContent = `AI: ${data.response}`;
        chatbox.appendChild(aiMessage);
        chatbox.scrollTop = chatbox.scrollHeight;  // Auto-scroll
    } catch (error) {
        console.error("Error:", error);
        let errorMessage = document.createElement("div");
        errorMessage.className = "message ai";
        errorMessage.textContent = "AI: Sorry, something went wrong!";
        chatbox.appendChild(errorMessage);
    }
}

// Allow sending message on "Enter" key press
function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}
