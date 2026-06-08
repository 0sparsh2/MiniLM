const chatContainer = document.getElementById('chatContainer');
const inputForm = document.getElementById('inputForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const statusIndicator = document.querySelector('.status-indicator');
const statusText = document.querySelector('.status-text');
const modelSelect = document.getElementById('modelSelect');

let ws;
let currentBotMessageEl = null;

const MODELS = {
    "arnir0/Tiny-LLM": {
        bin_path: "model_tiny_llm_q4.bin",
        vocab_path: "scripts/tiny_llm_vocab.txt"
    },
    "HuggingFaceTB/SmolLM-135M": {
        bin_path: "model_smollm_135m_q4.bin",
        vocab_path: "scripts/smollm_135m_vocab.txt"
    },
    "tinystories_v1": {
        bin_path: "",
        vocab_path: ""
    },
    "tinystories_v2": {
        bin_path: "",
        vocab_path: ""
    },
    "bitnet_instruct": {
        bin_path: "",
        vocab_path: ""
    },
    "bitnet_instruct_v5": {
        bin_path: "",
        vocab_path: ""
    }
};

function connect() {
    ws = new WebSocket(`ws://${window.location.host}`);

    ws.onopen = () => {
        statusIndicator.classList.add('connected');
        statusText.textContent = 'Connected';
        // Request model load on connect
        requestModelLoad();
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'loaded') {
                statusText.textContent = 'Ready';
                messageInput.disabled = false;
                sendBtn.disabled = false;
                messageInput.focus();
                appendSystemMessage("Model loaded and ready.");
            } else if (msg.type === 'progress') {
                statusText.textContent = msg.text;
                appendSystemMessage(msg.text);
            } else if (msg.type === 'token') {
                if (!currentBotMessageEl) {
                    currentBotMessageEl = createMessageElement('bot');
                    chatContainer.appendChild(currentBotMessageEl);
                }
                currentBotMessageEl.textContent += msg.text;
                scrollToBottom();
            } else if (msg.type === 'end') {
                currentBotMessageEl = null;
                messageInput.disabled = false;
                sendBtn.disabled = false;
                messageInput.focus();
            } else if (msg.type === 'error') {
                appendSystemMessage("Error: " + msg.error);
                messageInput.disabled = false;
                sendBtn.disabled = false;
            }
        } catch(e) {
            console.error("Invalid msg from server", event.data);
        }
    };

    ws.onclose = () => {
        disconnect();
        setTimeout(connect, 3000); // Auto-reconnect
    };
}

function requestModelLoad() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    statusText.textContent = 'Loading model...';
    messageInput.disabled = true;
    sendBtn.disabled = true;
    const model_id = modelSelect.value;
    ws.send(JSON.stringify({
        type: "load",
        model_id: model_id,
        bin_path: MODELS[model_id].bin_path,
        vocab_path: MODELS[model_id].vocab_path
    }));
}

modelSelect.addEventListener('change', () => {
    chatContainer.innerHTML = '';
    requestModelLoad();
});

function disconnect() {
    statusIndicator.classList.remove('connected');
    statusText.textContent = 'Disconnected';
    messageInput.disabled = true;
    sendBtn.disabled = true;
}

function appendUserMessage(text) {
    const el = createMessageElement('user');
    el.textContent = text;
    chatContainer.appendChild(el);
    scrollToBottom();
}

function appendSystemMessage(text) {
    const el = createMessageElement('bot');
    el.style.opacity = '0.7';
    el.style.fontStyle = 'italic';
    el.textContent = text;
    chatContainer.appendChild(el);
    scrollToBottom();
}

function createMessageElement(sender) {
    const el = document.createElement('div');
    el.classList.add('message', `${sender}-message`);
    return el;
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

inputForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    appendUserMessage(text);
    currentBotMessageEl = null;
    
    ws.send(JSON.stringify({
        type: "prompt",
        prompt: text
    }));
    
    messageInput.value = '';
    messageInput.disabled = true;
    sendBtn.disabled = true;
});

// Init
connect();
