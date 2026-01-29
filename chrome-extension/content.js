// DeFake Content Script

let overlay = null;
let snipperOverlay = null;
let startX, startY;

// Listen for messages
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "show_result") {
        showResult(request.type, request.data);
    } else if (request.action === "show_error") {
        showError(request.message);
    } else if (request.action === "start_snipping") {
        startSnippingTool();
    }
});

function createOverlay() {
    if (overlay) document.body.removeChild(overlay);

    overlay = document.createElement("div");
    overlay.className = "defake-overlay";

    overlay.innerHTML = `
    <div class="defake-header">
      <span class="defake-title">DeFake Analysis</span>
      <span class="defake-close">✕</span>
    </div>
    <div class="defake-content">
      <div class="defake-loading">Analyzing...</div>
    </div>
  `;

    overlay.querySelector(".defake-close").onclick = () => {
        document.body.removeChild(overlay);
        overlay = null;
    };

    document.body.appendChild(overlay);
}

function showResult(type, data) {
    if (!overlay) createOverlay();

    const contentDiv = overlay.querySelector(".defake-content");

    let verdictClass = "";
    let verdictText = "";
    let confidence = 0;

    if (type === "text") {
        verdictClass = data.label === "AI" ? "AI" : "HUMAN";
        verdictText = data.label === "AI" ? "AI GENERATED" : "HUMAN WRITTEN";
        confidence = data.confidence;
    } else {
        verdictClass = data.is_fake ? "FAKE" : "REAL";
        verdictText = data.is_fake ? "DEEPFAKE DETECTED" : "LIKELY REAL";
        confidence = data.confidence;
    }

    contentDiv.innerHTML = `
    <div class="defake-verdict ${verdictClass}">${verdictText}</div>
    <div class="defake-confidence">Confidence: ${confidence}%</div>
    ${type === 'text' && data.ai_probability ?
            `<div class="defake-details">AI Probability: ${(data.ai_probability * 100).toFixed(1)}%</div>` : ''}
  `;
}

function showError(msg) {
    if (!overlay) createOverlay();
    const contentDiv = overlay.querySelector(".defake-content");
    contentDiv.innerHTML = `<div style="color: #ef4444;">Error: ${msg}</div>`;
}

// --- Snipping Tool ---

function startSnippingTool() {
    document.body.style.cursor = "crosshair";

    snipperOverlay = document.createElement("div");
    snipperOverlay.className = "defake-snipper-overlay";
    document.body.appendChild(snipperOverlay);

    let selectionBox = document.createElement("div");
    selectionBox.className = "defake-selection-box";
    selectionBox.style.display = 'none';
    snipperOverlay.appendChild(selectionBox);

    let isDragging = false;

    snipperOverlay.addEventListener("mousedown", (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;

        selectionBox.style.left = startX + 'px';
        selectionBox.style.top = startY + 'px';
        selectionBox.style.width = '0px';
        selectionBox.style.height = '0px';
        selectionBox.style.display = 'block';
    });

    snipperOverlay.addEventListener("mousemove", (e) => {
        if (!isDragging) return;

        const currentX = e.clientX;
        const currentY = e.clientY;

        const width = Math.abs(currentX - startX);
        const height = Math.abs(currentY - startY);
        const left = Math.min(currentX, startX);
        const top = Math.min(currentY, startY);

        selectionBox.style.width = width + 'px';
        selectionBox.style.height = height + 'px';
        selectionBox.style.left = left + 'px';
        selectionBox.style.top = top + 'px';
    });

    snipperOverlay.addEventListener("mouseup", async (e) => {
        isDragging = false;
        document.body.style.cursor = "default";

        const rect = selectionBox.getBoundingClientRect();

        // Remove overlay immediately
        document.body.removeChild(snipperOverlay);
        snipperOverlay = null;

        if (rect.width < 10 || rect.height < 10) return; // Too small

        if (rect.width < 10 || rect.height < 10) return;

        try {
            chrome.runtime.sendMessage({
                action: "capture_area",
                rect: {
                    x: rect.x * window.devicePixelRatio,
                    y: rect.y * window.devicePixelRatio,
                    width: rect.width * window.devicePixelRatio,
                    height: rect.height * window.devicePixelRatio
                }
            });
        } catch (err) {
            console.error(err);
            showError("Failed to initiate capture");
        }
    });
}
