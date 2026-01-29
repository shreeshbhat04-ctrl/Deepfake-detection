//const API_URL = "http://localhost:8000"; // Update for prod
const API_URL = "https://shreesha1-deepfake.hf.space";

document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
        });
    });

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileNameDisplay = document.getElementById('file-name');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#8b5cf6';
        dropZone.style.backgroundColor = 'rgba(139, 92, 246, 0.1)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#334155';
        dropZone.style.backgroundColor = '#1e293b';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#334155';
        dropZone.style.backgroundColor = '#1e293b';

        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameDisplay.textContent = file.name;
        analyzeBtn.disabled = false;
        hideResult();
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        setLoading(true);
        hideResult();

        const formData = new FormData();
        formData.append("file", selectedFile);

        const endpoint = selectedFile.type.startsWith('image') ? '/analyze-image' : '/predict';

        try {
            const response = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            showResult(data);
        } catch (error) {
            console.error(error);
            showError("Analysis failed. Check server.");
        } finally {
            setLoading(false);
        }
    });

    const textInput = document.getElementById('text-input');
    const analyzeTextBtn = document.getElementById('analyze-text-btn');

    analyzeTextBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) return;

        setLoading(true);
        hideResult();

        try {
            const response = await fetch(`${API_URL}/analyze-text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();
            showTextResult(data);
        } catch (error) {
            console.error(error);
            showError("Text analysis failed.");
        } finally {
            setLoading(false);
        }
    });

    const startSnipBtn = document.getElementById('start-snip-btn');
    startSnipBtn.addEventListener('click', async () => {
        // Send message to active tab to start snipping
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (tab) {
            chrome.tabs.sendMessage(tab.id, { action: "start_snipping" });
            window.close(); // Close popup so user can snip
        }
    });

    function setLoading(isLoading) {
        const activeTab = document.querySelector('.tab.active').dataset.tab;
        const btn = activeTab === 'upload' ? analyzeBtn : analyzeTextBtn;

        if (isLoading) {
            btn.textContent = "Analyzing...";
            btn.disabled = true;
        } else {
            btn.textContent = activeTab === 'upload' ? "Analyze File" : "Analyze Text";
            btn.disabled = false;
        }
    }

    function hideResult() {
        document.getElementById('result-area').style.display = 'none';
    }

    function showResult(data) {
        const resultArea = document.getElementById('result-area');
        const verdict = document.getElementById('verdict');
        const confidence = document.getElementById('confidence');

        resultArea.style.display = 'block';

        const isFake = data.is_fake;
        const msg = isFake ? "DEEPFAKE DETECTED" : "LIKELY REAL";

        verdict.textContent = msg;
        verdict.className = "result-verdict " + (isFake ? "FAKE" : "REAL");

        confidence.textContent = `Confidence: ${data.confidence}%`;
    }

    function showTextResult(data) {
        const resultArea = document.getElementById('result-area');
        const verdict = document.getElementById('verdict');
        const confidence = document.getElementById('confidence');

        resultArea.style.display = 'block';

        const isAi = data.label === 'AI';
        const msg = isAi ? "AI GENERATED" : "HUMAN WRITTEN";

        verdict.textContent = msg;
        verdict.className = "result-verdict " + (isAi ? "FAKE" : "REAL");

        confidence.textContent = `Confidence: ${data.confidence}%`;
    }

    function showError(msg) {
        const resultArea = document.getElementById('result-area');
        const verdict = document.getElementById('verdict');
        const confidence = document.getElementById('confidence');

        resultArea.style.display = 'block';
        verdict.textContent = "ERROR";
        verdict.className = "result-verdict FAKE"; // Red color
        confidence.textContent = msg;
    }
});
