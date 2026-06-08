const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.static(path.join(__dirname, 'public')));

// Spawn the persistent python proxy
let proxyProcess = spawn('venv/bin/python3', ['scripts/server_proxy.py'], {
    cwd: path.join(__dirname, '..'),
});

proxyProcess.stdout.on('data', (data) => {
    // The python script outputs JSON lines
    const lines = data.toString().split('\n');
    for (const line of lines) {
        if (!line.trim()) continue;
        try {
            // Broadcast to all connected clients
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(line);
                }
            });
        } catch (e) {
            console.error("Error parsing/sending:", e);
        }
    }
});

proxyProcess.stderr.on('data', (data) => {
    console.error(`[Python] ${data.toString()}`);
});

wss.on('connection', (ws) => {
    console.log('Client connected');

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            // Expected format from client:
            // { model_id: "...", bin_path: "...", vocab_path: "...", prompt: "..." }
            
            // Forward directly to Python proxy
            proxyProcess.stdin.write(JSON.stringify(data) + '\n');
        } catch(e) {
            console.error("Invalid JSON from client:", message.toString());
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server listening on http://localhost:${PORT}`);
});
