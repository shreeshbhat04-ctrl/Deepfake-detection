import time
import requests
import numpy as np
import cv2
import os

API_URL = "http://localhost:8000"

def create_dummy_image(filename="bench_test.jpg"):
    # Create a simple image with a "face" (random noise + a slightly structured rect to hopefully trigger MTCNN)
    # Actually, MTCNN needs a real-ish face. Let's just create a blank image, 
    # if MTCNN fails it's fine, we want to measure the "overhead" mostly, 
    # but ideally we want the full pipeline.
    # For now, let's just make a black image. The server returns "Could not detect face" fast, 
    # so we might measure the "worst case" (full processing) if we had a real face.
    # But for a quick "system response" test, a blank image is a baseline (server still does I/O, loads model if needed).
    
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(filename, img)
    return filename

def benchmark_text():
    print("\n--- Benchmarking Text Detection (ModernBERT) ---")
    text = "This is a simple test sentence to check the speed of the model."
    
    # Warmup
    try:
        requests.post(f"{API_URL}/analyze-text", json={"text": text})
    except:
        print("Server seems down!")
        return

    times = []
    for i in range(5):
        start = time.time()
        response = requests.post(f"{API_URL}/analyze-text", json={"text": text})
        lat = time.time() - start
        times.append(lat)
        print(f"Request {i+1}: {lat:.4f}s")
    
    print(f"Average Text Latency: {sum(times)/len(times):.4f}s")

def benchmark_image():
    print("\n--- Benchmarking Image Pipeline (ResNeXt + MTCNN) ---")
    filename = create_dummy_image()
    
    # Warmup
    try:
        with open(filename, "rb") as f:
            requests.post(f"{API_URL}/analyze-image", files={"file": f})
    except:
        print("Server seems down!")
        return

    times = []
    for i in range(5):
        start = time.time()
        with open(filename, "rb") as f:
            requests.post(f"{API_URL}/analyze-image", files={"file": f})
        lat = time.time() - start
        times.append(lat)
        print(f"Request {i+1}: {lat:.4f}s")
    
    os.remove(filename)
    print(f"Average Image Latency: {sum(times)/len(times):.4f}s")

if __name__ == "__main__":
    print("Starting Benchmark...")
    print("Note: First request might include model loading time (if not warm).")
    benchmark_text()
    benchmark_image()
