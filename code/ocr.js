/**
  * This module creates a 200x200 pixel canvas for a user to draw
  * digits. The digits can either be used to train the neural network
  * or to test the network's current prediction for that digit.
  *
  * To simplify computation, the 200x200px canvas is translated as a 20x20px
  * canvas to be processed as an input array of 1s (white) and 0s (black) on
  * on the server side. Each new translated pixel's size is 10x10px
  *
  * When training the network, traffic to the server can be reduced by batching
  * requests to train based on BATCH_SIZE.
  */
var ocrDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH
    BATCH_SIZE: 1,

    // Auto-detect environment: use local server when running locally, /api/ on Vercel
    API_URL: (window.location.hostname === 'localhost' ||
              window.location.hostname === '127.0.0.1' ||
              window.location.hostname === '')   // file:// opened directly
        ? 'http://localhost:8000'
        : '/api/',

    // Visual Colors (Data model remains 0=bg, 1=stroke)
    BG_COLOR: "#12172B",     // --canvas-bg
    STROKE_COLOR: "#7C8CF0", // Brightened --signal for dark canvas
    GRID_COLOR: "rgba(76, 95, 213, 0.25)", // Low-opacity --signal

    trainArray: [],
    trainingRequestCount: 0,
    isTrainInProgress: false,

    onLoadFunction: function () {
        this.resetCanvas();
        this.logActivity("System initialized. Ready for input.");
        this.pingServer();
    },

    pingServer: function () {
        var self = this;
        var xhr = new XMLHttpRequest();
        xhr.open('GET', this.API_URL, true);
        xhr.onload = function () {
            if (xhr.status === 200) {
                self.updateConnectionStatus("connected");
                self.logActivity("Server connected.", "success");
            } else {
                self.updateConnectionStatus("offline");
            }
        };
        xhr.onerror = function () { self.updateConnectionStatus("offline"); };
        xhr.send();
    },

    resetCanvas: function () {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');

        this.data = [];
        ctx.fillStyle = this.BG_COLOR;
        ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
        var matrixSize = 400;
        while (matrixSize--) this.data.push(0);
        this.drawGrid(ctx);

        canvas.onmousemove = function (e) { this.onMouseMove(e, ctx, canvas) }.bind(this);
        canvas.onmousedown = function (e) { this.onMouseDown(e, ctx, canvas) }.bind(this);
        canvas.onmouseup = function (e) { this.onMouseUp(e, canvas) }.bind(this);

        this.resetNetworkOutput();
    },

    drawGrid: function (ctx) {
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.GRID_COLOR;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    onMouseMove: function (e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown: function (e, ctx, canvas) {
        canvas.isDrawing = true;
        canvas.parentElement.classList.add('drawing');
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseUp: function (e, canvas) {
        canvas.isDrawing = false;
        canvas.parentElement.classList.remove('drawing');
    },

    fillSquare: function (ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        if (xPixel < 0 || xPixel >= this.TRANSLATED_WIDTH || yPixel < 0 || yPixel >= this.TRANSLATED_WIDTH) {
            return;
        }
        // Underlying data model expects 1 for ink, 0 for background
        this.data[((xPixel - 1) * this.TRANSLATED_WIDTH + yPixel) - 1] = 1;

        ctx.fillStyle = this.STROKE_COLOR;
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, this.PIXEL_WIDTH, this.PIXEL_WIDTH);
    },

    train: function () {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            this.logActivity("Validation failed: Please type and draw a digit value in order to train.", "error");
            return;
        }
        this.trainArray.push({ "y0": this.data, "label": parseInt(digitVal) });
        this.trainingRequestCount++;

        // Time to send a training batch to the server.
        if (this.trainingRequestCount == this.BATCH_SIZE) {
            this.logActivity("Sending training data to server for digit: " + digitVal);
            var json = {
                trainArray: this.trainArray,
                train: true
            };
            this.isTrainInProgress = true;
            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }
    },

    test: function () {
        if (this.data.indexOf(1) < 0) {
            this.logActivity("Validation failed: Please draw a digit in order to test.", "error");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.isTrainInProgress = false;
        this.sendData(json);
    },

    receiveResponse: function (xmlHttp) {
        if (xmlHttp.status != 200) {
            this.updateConnectionStatus("offline");
            var errorMsg = xmlHttp.statusText;
            try {
                var responseJSON = JSON.parse(xmlHttp.responseText);
                if (responseJSON.message) errorMsg = responseJSON.message;
            } catch (e) { }
            this.logActivity("Server Error (" + xmlHttp.status + "): " + errorMsg, "error");
            return;
        }

        this.updateConnectionStatus("connected");
        if (xmlHttp.responseText) {
            try {
                var responseJSON = JSON.parse(xmlHttp.responseText);
                if (responseJSON.type == "test") {
                    var result = responseJSON.result;
                    this.logActivity("Predicted: " + result, "success");
                    this.updateNetworkOutput(result);
                } else if (this.isTrainInProgress) {
                    this.logActivity("Training successful. Weights saved.", "success");
                }
            } catch (e) {
                this.logActivity("Failed to parse response: " + xmlHttp.responseText, "error");
            }
        } else {
            if (this.isTrainInProgress) {
                this.logActivity("Training successful. Weights saved.", "success");
            }
        }
        this.isTrainInProgress = false;
    },

    onError: function (xmlHttp) {
        this.updateConnectionStatus("offline");
        this.logActivity("Error occurred while connecting to server: " + xmlHttp.statusText, "error");
    },

    sendData: function (json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', this.API_URL, true);
        xmlHttp.onload = function () { this.receiveResponse(xmlHttp); }.bind(this);
        xmlHttp.onerror = function () { this.onError(xmlHttp) }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-length', msg.length);
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.send(msg);
    },

    logActivity: function (message, type) {
        var logContainer = document.getElementById("activity-log");
        if (!logContainer) return;

        var entry = document.createElement("div");
        entry.className = "log-entry" + (type ? " " + type : "");

        var timeSpan = document.createElement("span");
        timeSpan.className = "log-time";
        var now = new Date();
        timeSpan.textContent = now.getHours().toString().padStart(2, '0') + ":" +
            now.getMinutes().toString().padStart(2, '0') + ":" +
            now.getSeconds().toString().padStart(2, '0');

        var msgSpan = document.createElement("span");
        msgSpan.className = "log-msg";
        msgSpan.textContent = message;

        entry.appendChild(timeSpan);
        entry.appendChild(msgSpan);

        logContainer.insertBefore(entry, logContainer.firstChild);
    },

    resetNetworkOutput: function () {
        for (var i = 0; i <= 9; i++) {
            var row = document.getElementById("bar-row-" + i);
            var fill = document.getElementById("bar-fill-" + i);
            if (row) {
                row.classList.remove("winner");
                row.classList.remove("active");
            }
            if (fill) fill.style.width = "0%";
        }
        var label = document.getElementById("predicted-winner");
        if (label) label.textContent = "";
    },

    updateNetworkOutput: function (winningDigit) {
        this.resetNetworkOutput();

        var label = document.getElementById("predicted-winner");
        if (label) label.textContent = "Predicted: " + winningDigit;

        for (var i = 0; i <= 9; i++) {
            var row = document.getElementById("bar-row-" + i);
            var fill = document.getElementById("bar-fill-" + i);
            if (row) row.classList.add("active");

            if (i === winningDigit) {
                if (row) row.classList.add("winner");
                if (fill) fill.style.width = "100%";
            } else {
                if (fill) fill.style.width = "0%";
            }
        }
    },

    updateConnectionStatus: function (status) {
        var badge = document.getElementById("connection-status");
        if (!badge) return;
        if (status === "connected") {
            badge.className = "status-badge connected";
            badge.innerHTML = '<span class="status-dot"></span>Server connected';
        } else {
            badge.className = "status-badge offline";
            badge.innerHTML = '<span class="status-dot"></span>Offline';
        }
    }
}