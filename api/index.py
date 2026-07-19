"""
Neural-OCR Vercel Serverless Handler
Self-contained: OCRNeuralNetwork class is embedded directly so there are
no import-path issues in Vercel's runtime environment.
"""
import json
import os
import math
from http.server import BaseHTTPRequestHandler

import numpy as np

# ── Embedded OCRNeuralNetwork (from code/ocr.py) ──────────────────────────────
_NN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn.json")


class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    WIDTH_IN_PIXELS = 20
    NN_FILE_PATH = _NN_FILE

    def __init__(self, num_hidden_nodes, data_matrix, data_labels,
                 training_indices, use_file=True):
        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        self.sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)
        self._use_file = use_file
        self.data_matrix = data_matrix
        self.data_labels = data_labels

        if not os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH) or not use_file:
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, 10)
            if training_indices:
                from collections import namedtuple
                TrainData = namedtuple('TrainData', ['y0', 'label'])
                self.train([TrainData(self.data_matrix[i],
                                     int(self.data_labels[i]))
                            for i in training_indices])
            self.save()
        else:
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06)
                for x in np.random.rand(size_out, size_in)]

    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)

    def _sigmoid_prime_scalar(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def train(self, training_data_array):
        for data in training_data_array:
            y0 = data.y0 if hasattr(data, 'y0') else data['y0']
            label = data.label if hasattr(data, 'label') else data['label']

            y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(y0).T)
            sum1 = y1 + np.asmatrix(self.input_layer_bias)
            y1 = self.sigmoid(sum1)

            y2 = np.dot(np.array(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias)
            y2 = self.sigmoid(y2)

            actual_vals = [0] * 10
            actual_vals[label] = 1
            output_errors = np.asmatrix(actual_vals).T - np.asmatrix(y2)
            hidden_errors = np.multiply(
                np.dot(np.asmatrix(self.theta2).T, output_errors),
                self.sigmoid_prime(sum1))

            self.theta1 += self.LEARNING_RATE * np.dot(
                np.asmatrix(hidden_errors), np.asmatrix(y0))
            self.theta2 += self.LEARNING_RATE * np.dot(
                np.asmatrix(output_errors), np.asmatrix(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(test).T)
        y1 = y1 + np.asmatrix(self.input_layer_bias)
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return
        payload = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0],
            "b2": self.hidden_layer_bias[0].tolist()[0],
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as f:
            json.dump(payload, f)

    def _load(self):
        with open(OCRNeuralNetwork.NN_FILE_PATH) as f:
            data = json.load(f)
        self.theta1 = [np.array(li) for li in data['theta1']]
        self.theta2 = [np.array(li) for li in data['theta2']]
        self.input_layer_bias = [np.array(data['b1'][0])]
        self.hidden_layer_bias = [np.array(data['b2'][0])]


# ── Initialise neural network ─────────────────────────────────────────────────
try:
    nn = OCRNeuralNetwork(15, [], [], [], use_file=True)
    print("[api/index.py] Neural network loaded from", _NN_FILE, flush=True)
except Exception as _e:
    print("[api/index.py] NN init FAILED:", _e, flush=True)
    nn = None


# ── Vercel handler ────────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        status = "ready" if nn else "error: NN not loaded"
        self.wfile.write(json.dumps({"status": status}).encode())

    def do_POST(self):
        code = 200
        resp = {}
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))

            if payload.get("train"):
                bad = [
                    item["label"] for item in payload["trainArray"]
                    if not isinstance(item["label"], int)
                    or not (0 <= item["label"] <= 9)
                ]
                if bad:
                    code, resp = 400, {"type": "error",
                                       "message": "Labels must be 0-9."}
                elif nn is None:
                    code, resp = 500, {"type": "error",
                                       "message": "NN not loaded."}
                else:
                    nn.train(payload["trainArray"])
                    try:
                        nn.save()
                    except Exception:
                        pass          # read-only filesystem on Vercel — ignore
                    resp = {"type": "train", "message": "Training successful."}

            elif payload.get("predict"):
                if nn is None:
                    code, resp = 500, {"type": "error", "message": "NN not loaded."}
                else:
                    resp = {"type": "test",
                            "result": nn.predict(payload["image"])}
            else:
                code, resp = 400, {"type": "error", "message": "Unknown action."}

        except Exception as exc:
            code, resp = 500, {"type": "error", "message": str(exc)}

        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Content-Length, Connection")
