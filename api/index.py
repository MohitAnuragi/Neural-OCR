import json
import os
import sys
import numpy as np
from http.server import BaseHTTPRequestHandler

# ── path setup ────────────────────────────────────────────────────────────────
# Make the sibling "code" directory importable so we can use ocr.py
_HERE = os.path.dirname(__file__)          # .../api/
_CODE = os.path.join(_HERE, "..", "code")  # .../code/
sys.path.insert(0, os.path.abspath(_CODE))

from ocr import OCRNeuralNetwork

# ── initialise the neural network once (serverless warm start) ─────────────────
HIDDEN_NODE_COUNT = 15

try:
    nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, [], [], [], use_file=True)
except Exception as e:
    print(f"[api/index.py] NN init error: {e}", flush=True)
    nn = None

# ── Vercel Python handler (must be called `handler`) ──────────────────────────
class handler(BaseHTTPRequestHandler):

    # ── CORS preflight ────────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ── Health-check / browser GET ────────────────────────────────────────────
    def do_GET(self):
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "Neural-OCR API running"}).encode())

    # ── Main API endpoint ─────────────────────────────────────────────────────
    def do_POST(self):
        response_code = 200
        response = {}
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body)

            if payload.get("train"):
                invalid = [
                    item["label"]
                    for item in payload["trainArray"]
                    if not isinstance(item["label"], int)
                    or not (0 <= item["label"] <= 9)
                ]
                if invalid:
                    response_code = 400
                    response = {"type": "error", "message": "Labels must be integers 0-9."}
                elif nn is None:
                    response_code = 500
                    response = {"type": "error", "message": "Neural network not initialised."}
                else:
                    nn.train(payload["trainArray"])
                    # Vercel filesystem is read-only — skip save silently
                    try:
                        nn.save()
                    except Exception:
                        pass
                    response = {"type": "train", "message": "Training successful."}

            elif payload.get("predict"):
                if nn is None:
                    response_code = 500
                    response = {"type": "error", "message": "Neural network not initialised."}
                else:
                    result = nn.predict(payload["image"])
                    response = {"type": "test", "result": result}
            else:
                response_code = 400
                response = {"type": "error", "message": "Unknown action."}

        except Exception as exc:
            response_code = 500
            response = {"type": "error", "message": str(exc)}

        self.send_response(response_code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    # ── helper ────────────────────────────────────────────────────────────────
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Content-Length, Connection")
