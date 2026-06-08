/**
 * sparse_server.js — Node.js WebSocket bridge for Sparse 2:4 BitNet chat.
 * Serves the UI at http://localhost:3333
 * Bridges browser WebSocket ↔ Python sparse_chat_server.py subprocess.
 */

const express   = require('express');
const http      = require('http');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const path      = require('path');

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({ server });
const PORT   = 3333;

// ── Serve static files from ui/public ────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));

// ── Redirect root → sparse_chat.html ─────────────────────────────────────────
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'sparse_chat.html'));
});

// ── Spawn Python model process ────────────────────────────────────────────────
const PYTHON = path.join(__dirname, '..', '.venv', 'bin', 'python3');
const SCRIPT = path.join(__dirname, '..', 'scripts', 'sparse_chat_server.py');

console.log(`Spawning: ${PYTHON} ${SCRIPT}`);
const py = spawn(PYTHON, [SCRIPT], {
    cwd: path.join(__dirname, '..'),
    stdio: ['pipe', 'pipe', 'pipe'],
});

py.stderr.on('data', d => console.log(`[model] ${d.toString().trim()}`));

// Track connected WebSocket clients
const clients = new Set();

let modelReady = false;
let pendingBuffer = '';

// ── Parse JSON lines from Python stdout ───────────────────────────────────────
py.stdout.on('data', chunk => {
    pendingBuffer += chunk.toString();
    const lines = pendingBuffer.split('\n');
    pendingBuffer = lines.pop(); // keep incomplete last line

    for (const line of lines) {
        if (!line.trim()) continue;
        try {
            const msg = JSON.parse(line);

            if (msg.ready) {
                modelReady = true;
                console.log('✓ Model ready');
                broadcast({ type: 'model_ready' });
                return;
            }

            // Forward token/done messages to the matching client
            broadcast({ type: 'token', ...msg });

        } catch (e) {
            console.error('Bad JSON from python:', line);
        }
    }
});

py.on('exit', code => {
    console.error(`Python process exited with code ${code}`);
    broadcast({ type: 'error', message: 'Model process died. Please restart the server.' });
});

function broadcast(obj) {
    const str = JSON.stringify(obj);
    for (const ws of clients) {
        if (ws.readyState === WebSocket.OPEN) ws.send(str);
    }
}

// ── WebSocket handler ─────────────────────────────────────────────────────────
wss.on('connection', ws => {
    clients.add(ws);
    console.log(`Client connected (${clients.size} total)`);

    // Tell client whether model is ready
    ws.send(JSON.stringify({ type: modelReady ? 'model_ready' : 'model_loading' }));

    ws.on('message', raw => {
        try {
            const { prompt, id, max_tokens } = JSON.parse(raw);
            if (!modelReady) {
                ws.send(JSON.stringify({ type: 'error', message: 'Model still loading…' }));
                return;
            }
            if (!prompt || !prompt.trim()) return;
            // Forward to Python
            py.stdin.write(JSON.stringify({ prompt: prompt.trim(), id, max_tokens: max_tokens || 200 }) + '\n');
        } catch (e) {
            console.error('Invalid message from client:', raw.toString());
        }
    });

    ws.on('close', () => {
        clients.delete(ws);
        console.log(`Client disconnected (${clients.size} remaining)`);
    });
});

server.listen(PORT, () => {
    console.log(`\n🚀  Sparse 2:4 Chat UI → http://localhost:${PORT}/\n`);
});
