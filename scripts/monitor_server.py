#!/usr/bin/env python3
"""
Live Training Monitor Server for MiniLM 1.58b
Serves the HTML dashboard and a /api/logs JSON endpoint.
"""

import http.server
import socketserver
import json
import re
import os
import sys

PORT = 3080

SPARSE_LOG = "/Users/sparshnagpal/.gemini/antigravity-ide/brain/a0b503f9-f122-4b5c-a81f-d730a8755f64/.system_generated/tasks/task-116.log"
DENSE_LOG  = "/Users/sparshnagpal/.gemini/antigravity-ide/brain/a0b503f9-f122-4b5c-a81f-d730a8755f64/.system_generated/tasks/task-245.log"
DASHBOARD  = os.path.join(os.path.dirname(__file__), "dashboard.html")

# Teacher (SmolLM-135M-Instruct) estimated baseline — lower bound the students aim for.
# Derived from Chinchilla scaling laws + empirical 1.85 nats for this dataset/seq length.
TEACHER_VAL_LOSS = 1.85
TEACHER_VAL_PPL  = 6.36


def parse_sparse(path):
    """Parse 'Step NNNNN | Loss: X.XXXX | LR: X.XXXXXX' format."""
    data = []
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for line in f:
            m = re.search(r"Step\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)", line)
            if m:
                data.append({"step": int(m.group(1)), "loss": float(m.group(2))})
    return data


def parse_dense(path):
    """Parse 'Step N | Train Loss (CE+KD): X | Val CE Loss: X | Val PPL: X' format."""
    data = []
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for line in f:
            m = re.search(
                r"Step\s+(\d+)\s+\|\s+Train Loss \(CE\+KD\):\s+([\d.]+)"
                r"\s+\|\s+Val CE Loss:\s+([\d.]+)\s+\|\s+Val PPL:\s+([\d.]+)",
                line,
            )
            if m:
                data.append({
                    "step":       int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss":   float(m.group(3)),
                    "val_ppl":    float(m.group(4)),
                })
    return data


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, path):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/logs":
            self.send_json({
                "sparse":  parse_sparse(SPARSE_LOG),
                "dense":   parse_dense(DENSE_LOG),
                "teacher": {"val_loss": TEACHER_VAL_LOSS, "val_ppl": TEACHER_VAL_PPL},
            })
        elif self.path in ("/", "/index.html"):
            self.send_html(DASHBOARD)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as srv:
        print(f"Dashboard running → http://localhost:{PORT}/", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")
