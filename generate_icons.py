from PIL import Image, ImageDraw
import os

def create_icon(size, path):
    img = Image.new('RGB', (size, size), color=(73, 109, 219)) # Primary color blue
    d = ImageDraw.Draw(img)
    d.text((size//4, size//4), "DF", fill=(255, 255, 255))
    img.save(path)
    print(f"Created {path}")

os.makedirs('chrome-extension/icons', exist_ok=True)
create_icon(16, 'chrome-extension/icons/icon16.png')
create_icon(48, 'chrome-extension/icons/icon48.png')
create_icon(128, 'chrome-extension/icons/icon128.png')
