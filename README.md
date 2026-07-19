# neural-ocr

![neural-ocr interface demo](https://img.shields.io/badge/status-active-success.svg)
![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-2.4+-orange.svg)

> A from-scratch neural network trained to read handwritten digits, featuring a dynamic visual interface.

**neural-ocr** is a lightweight, educational implementation of a Multi-Layer Perceptron (MLP) built entirely in Python using only NumPy. It operates without heavyweight ML frameworks like TensorFlow or PyTorch, making it an excellent demonstration of how forward-propagation and back-propagation algorithms work under the hood. 

The project includes a modern, responsive web interface that allows you to draw digits, feed them to the neural network for real-time predictions, and even train the network dynamically in your browser.

## Features

- **From-Scratch MLP:** A fully functional artificial neural network implemented with pure matrix math (NumPy).
- **Interactive UI:** A dark "instrument panel" drawing canvas paired with a live horizontal bar chart that visualizes the network's output layer.
- **Real-Time Training:** Draw a digit, supply the correct label, and click Train to actively run back-propagation and improve the network's local weights (`nn.json`).
- **Activity Log:** A built-in terminal-style log that tracks predictions, training events, and server status.
- **Zero-Dependency Frontend:** The UI is built with vanilla HTML, CSS, and JavaScript—no bundlers or frameworks required.

## Tech Stack

- **Backend:** Python 3, `http.server` (Standard Library)
- **Math/ML:** `numpy`
- **Frontend:** Vanilla HTML, CSS, JS

## Getting Started

### Prerequisites
- Python 3.x installed on your machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/neural-ocr.git
   cd neural-ocr
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy
   ```

### Running the Application

1. Open your terminal, activate your virtual environment, and navigate to the `code` directory:
   ```bash
   cd code
   python server.py
   ```
   *(The backend server will start silently on port `8000`)*

2. Open the frontend interface in your browser:
   Navigate to the `code` folder in your file explorer and double-click `ocr.html`.

## How to Use

### Testing the Network (Predicting)
1. Draw a single digit (0-9) on the dark canvas using your mouse.
2. Click the **Test** button.
3. The server will process the 20x20 pixel grid and return its prediction, which will be highlighted in the **Network Output** panel on the right.

### Training the Network (Learning)
If the network makes a mistake, you can correct it:
1. Draw the digit on the canvas.
2. Enter the correct number (0-9) in the **Digit** input box.
3. Click the **Train** button.
4. The backend will perform a back-propagation pass, adjust its internal weights, and save the updated model state to `nn.json`.

## Project Structure

```text
neural-ocr/
├── code/
│   ├── ocr.py                  # Core Neural Network logic (Forward/Back prop)
│   ├── server.py               # Lightweight HTTP server handling REST requests
│   ├── neural_network_design.py# Network architecture experiments/design script
│   ├── ocr.html                # Frontend UI presentation layer
│   ├── ocr.css                 # Design system and UI styling
│   ├── ocr.js                  # Canvas capture and XHR logic
│   ├── data.csv                # Initial raw training dataset
│   ├── dataLabels.csv          # Initial training dataset labels
│   └── nn.json                 # Serialized weight matrices (generated after training)
└── README.md
```

## How It Works

The network translates the 200x200 pixel HTML5 canvas into a downsampled 20x20 binary matrix (400 input nodes). 

1. **Input Layer:** 400 nodes representing the downsampled pixel grid (0 for background, 1 for ink).
2. **Hidden Layer:** Processed via a custom Sigmoid activation function.
3. **Output Layer:** 10 nodes representing the digits 0-9. The node with the highest activation value determines the network's prediction.
