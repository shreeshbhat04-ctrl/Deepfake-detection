import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# Import Facenet-PyTorch for Face Detection (No TensorFlow needed)
from facenet_pytorch import MTCNN

# --- 1. CONFIGURATION ---
# 10 frames is enough for a resume project and runs faster on CPU
SEQUENCE_LENGTH_DEFAULT = 10 
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. INITIALIZE MTCNN ---
print(f"Initializing MTCNN on {DEVICE}...")
# keep_all=True returns all faces, we'll sort them. 
# select_largest=False because we manually sort by confidence/size if needed, but 'keep_all=False' (default) returns only best face? 
# actually detect returns all.
mtcnn_detector = MTCNN(keep_all=True, device=DEVICE)

# Standard normalization
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. PREPROCESSING FUNCTION ---
def extract_frames_from_video(video_path, sequence_length=SEQUENCE_LENGTH_DEFAULT):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
    processed_frames = []
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue

        # Convert to RGB for MTCNN (OpenCV is BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces
            # boxes given as [x1, y1, x2, y2]
            boxes, probs = mtcnn_detector.detect(frame_rgb)
            
            if boxes is not None and len(boxes) > 0:
                # Get highest probability face or first one?
                # probs is list of probabilities. Filter valid ones.
                # Just take the one with standard highest probability.
                
                # Combine boxes and probs to sort
                face_list = []
                for box, prob in zip(boxes, probs):
                    if prob is None: continue
                    face_list.append({'box': box, 'conf': prob})
                
                if not face_list: continue

                best_face = sorted(face_list, key=lambda x: x['conf'], reverse=True)[0]
                x1, y1, x2, y2 = best_face['box']
                
                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1
                
                # Fix negative coordinates and float
                x, y = max(0, int(x)), max(0, int(y))
                w, h = int(w), int(h)
                
                # Add padding (10%)
                pad_w = int(w * 0.1)
                pad_h = int(h * 0.1)
                
                img_h, img_w, _ = frame.shape
                y_min = max(0, y - pad_h)
                y_max = min(img_h, y + h + pad_h)
                x_min = max(0, x - pad_w)
                x_max = min(img_w, x + w + pad_w)
                
                face_crop = frame[y_min:y_max, x_min:x_max]
                
                if face_crop.size != 0:
                    processed_frame = data_transforms(face_crop)
                    processed_frames.append(processed_frame)
        except Exception as e:
            # print(f"Frame processing error: {e}")
            continue

    cap.release()

    if not processed_frames:
        return None
    
    # Padding if we missed some frames due to detection failure
    while len(processed_frames) < sequence_length:
        processed_frames.append(processed_frames[-1])

    return torch.stack(processed_frames[:sequence_length])


# --- 3b. IMAGE PROCESSING FUNCTION ---
def process_image(image_path, sequence_length=SEQUENCE_LENGTH_DEFAULT):
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = mtcnn_detector.detect(frame_rgb)
        
        if boxes is None or len(boxes) == 0:
            return None

        face_list = []
        for box, prob in zip(boxes, probs):
            if prob is None: continue
            face_list.append({'box': box, 'conf': prob})
        
        if not face_list: return None

        best_face = sorted(face_list, key=lambda x: x['conf'], reverse=True)[0]
        x1, y1, x2, y2 = best_face['box']
        
        w = x2 - x1
        h = y2 - y1
        x = x1
        y = y1
        
        # Integer conversion and padding
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        
        pad_w = int(w * 0.1)
        pad_h = int(h * 0.1)
        
        img_h, img_w, _ = frame.shape
        y_min = max(0, y - pad_h)
        y_max = min(img_h, y + h + pad_h)
        x_min = max(0, x - pad_w)
        x_max = min(img_w, x + w + pad_w)
        
        face_crop = frame[y_min:y_max, x_min:x_max]
        
        if face_crop.size == 0:
            return None

        processed_frame = data_transforms(face_crop) # [3, 224, 224]
        
        # Repeat this frame to create a fake sequence
        return processed_frame.unsqueeze(0).repeat(sequence_length, 1, 1, 1)

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# --- 4. DATASET CLASS ---
class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, sequence_length=SEQUENCE_LENGTH_DEFAULT):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.video_files = []
        self.labels = []

        print(f" Scanning for videos in {data_dir}...")

        def find_videos_in_folder(folder_path):
            video_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_paths.append(os.path.join(root, file))
            return video_paths

        # --- 1. REAL VIDEOS (Limit 400) ---
        real_path = os.path.join(data_dir, 'real')
        real_videos = find_videos_in_folder(real_path)
        
        if len(real_videos) > 400:
            real_videos = real_videos[:400]

        for vid in real_videos:
            self.video_files.append(vid)
            self.labels.append(0)

        # --- 2. FAKE VIDEOS (Limit 400) ---
        fake_path = os.path.join(data_dir, 'fake')
        fake_videos = find_videos_in_folder(fake_path)
        
        if len(fake_videos) > 400:
            fake_videos = fake_videos[:400]

        for vid in fake_videos:
            self.video_files.append(vid)
            self.labels.append(1)

        self.total_videos = len(self.video_files)
        print(f" Total dataset size: {self.total_videos} videos")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        frames = extract_frames_from_video(video_path, self.sequence_length)
        
        if frames is None:
            return torch.zeros((self.sequence_length, 3, IMG_SIZE, IMG_SIZE)), -1 

        return frames, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    ds = DeepfakeDataset('data/')