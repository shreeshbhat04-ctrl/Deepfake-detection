// Background Service Worker

// Constants
//const API_URL = "http://localhost:8000";
const API_URL = "https://shreesha1-deepfake.hf.space";
// Create Context Menus
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "analyze-text",
        title: "DeFake: Analyze Selected Text",
        contexts: ["selection"]
    });

    chrome.contextMenus.create({
        id: "analyze-image",
        title: "DeFake: Analyze Image",
        contexts: ["image"]
    });
});

// Handle Clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "analyze-text") {
        analyzeText(info.selectionText, tab.id);
    } else if (info.menuItemId === "analyze-image") {
        analyzeImageFromUrl(info.srcUrl, tab.id);
    }
});

// Message Listener
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyze_text") {
        analyzeText(request.text, sender.tab.id, true);
    } else if (request.action === "analyze_screenshot") {
        analyzeScreenshot(request.imageData, sender.tab.id);
    } else if (request.action === "capture_area") {
        handleCaptureArea(request.rect, sender.tab.id);
    }
    return true;
});

// --- Helpers ---

async function handleCaptureArea(rect, tabId) {
    try {
        const dataUrl = await chrome.tabs.captureVisibleTab(null, { format: "png" });
        const croppedBlob = await cropImage(dataUrl, rect);

        if (croppedBlob) {
            analyzeBlob(croppedBlob, tabId, "snippet.png");
        } else {
            chrome.tabs.sendMessage(tabId, {
                action: "show_error",
                message: "Failed to process screenshot."
            });
        }
    } catch (e) {
        console.error(e);
        chrome.tabs.sendMessage(tabId, {
            action: "show_error",
            message: "Capture failed: " + e.message
        });
    }
}

async function cropImage(dataUrl, rect) {
    try {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob);

        const canvas = new OffscreenCanvas(rect.width, rect.height);
        const ctx = canvas.getContext('2d');

        ctx.drawImage(
            bitmap,
            rect.x, rect.y, rect.width, rect.height,
            0, 0, rect.width, rect.height
        );

        return await canvas.convertToBlob({ type: 'image/png' });
    } catch (e) {
        console.error("Cropping error:", e);
        return null;
    }
}

async function analyzeText(text, tabId, isContentScript = false) {
    try {
        if (!isContentScript) {
            chrome.scripting.executeScript({
                target: { tabId: tabId },
                function: () => alert("DeFake: Analyzing text...")
            });
        }

        const response = await fetch(`${API_URL}/analyze-text`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        chrome.tabs.sendMessage(tabId, {
            action: "show_result",
            type: "text",
            data: data
        });

    } catch (error) {
        console.error("Text analysis error:", error);
        chrome.tabs.sendMessage(tabId, {
            action: "show_error",
            message: "Could not connect to server."
        });
    }
}

async function analyzeImageFromUrl(imageUrl, tabId) {
    try {
        const res = await fetch(imageUrl);
        const blob = await res.blob();
        analyzeBlob(blob, tabId, "image.jpg");
    } catch (e) {
        console.error(e);
        chrome.tabs.sendMessage(tabId, {
            action: "show_error",
            message: "Could not download image."
        });
    }
}

async function analyzeScreenshot(dataUrl, tabId) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    analyzeBlob(blob, tabId, "screenshot.png");
}

async function analyzeBlob(blob, tabId, filename) {
    try {
        const formData = new FormData();
        formData.append("file", blob, filename);

        const response = await fetch(`${API_URL}/analyze-image`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        chrome.tabs.sendMessage(tabId, {
            action: "show_result",
            type: "image",
            data: data
        });

    } catch (error) {
        console.error("Image analysis error:", error);
        chrome.tabs.sendMessage(tabId, {
            action: "show_error",
            message: "Error analyzing image."
        });
    }
}
