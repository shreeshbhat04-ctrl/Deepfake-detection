import requests
from PIL import Image
import io
import os

# Create a dummy image
img = Image.new('RGB', (224, 224), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)

url = "http://localhost:8000/analyze-image"
files = {'file': ('test_image.jpg', img_byte_arr, 'image/jpeg')}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print(" Success!")
        print(response.json())
    else:
        print(f" Failed with status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f" Error: {e}")
    print("Make sure the backend is running!")
