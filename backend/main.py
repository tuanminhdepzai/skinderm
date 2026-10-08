from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
import datetime
import asyncio
import uuid
import requests
import json
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import timm
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io
import os
from dotenv import load_dotenv
from firebase_admin_utils import firebase_auth_manager
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
from explore_agent import run_explore_workflow, get_latest_blogs
import threading

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Skinderm AI Backend API", version="1.0.0")

# --- NVIDIA AI Client ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
nv_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY if NVIDIA_API_KEY else "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",
  timeout = 30.0,
  max_retries = 0
)

# Patch for imagekitio Pydantic forward ref resolution
try:
    import imagekitio.types.shared.overlay as _overlay
    import imagekitio.types.shared.src_options as _src_options
    _src_options.Overlay = _overlay.Overlay
except Exception:
    pass

from imagekitio import ImageKit

IMAGEKIT_PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY", "public_eilUB6xKm53YlRbH/rrMSl30xtg=")
IMAGEKIT_PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY", "private_SUBNq7mziNujilOpU2IykJNiXSo=")
IMAGEKIT_URL_ENDPOINT = os.getenv("IMAGEKIT_URL_ENDPOINT", "https://ik.imagekit.io/codebykdvn")

# Khởi tạo ImageKit client (SDK v5 nhận private_key)
try:
    imagekit = ImageKit(private_key=IMAGEKIT_PRIVATE_KEY)
except Exception:
    imagekit = None

def upload_to_imagekit(file_bytes: bytes, file_name: str, folder: str = "/scans") -> str:
    """Upload tệp byte lên ImageKit Cloud và trả về CDN URL công khai."""
    if not imagekit:
        return ""
    try:
        if hasattr(imagekit, "files") and hasattr(imagekit.files, "upload"):
            res = imagekit.files.upload(
                file=file_bytes,
                file_name=file_name,
                folder=folder
            )
        elif hasattr(imagekit, "upload_file"):
            res = imagekit.upload_file(
                file=file_bytes,
                file_name=file_name,
                options={"folder": folder}
            )
        else:
            res = None

        if res:
            url = getattr(res, "url", None)
            if not url and isinstance(res, dict):
                url = res.get("url")
            if not url and hasattr(res, "response_metadata"):
                raw = getattr(res.response_metadata, "raw", {})
                if isinstance(raw, dict):
                    url = raw.get("url")
            if url:
                if not url.startswith("http"):
                    endpoint = IMAGEKIT_URL_ENDPOINT.rstrip("/")
                    url = f"{endpoint}/{url.lstrip('/')}"
                return url
        return ""
    except Exception as e:
        print(f"[ERROR] ImageKit upload error: {e}")
        return ""


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. DIAGNOSIS ENSEMBLE MODEL (efficientnet_b3 + convnext_tiny) -> 8 classes
# ---------------------------------------------------------
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EnsembleModel, self).__init__()
        self.backbone1 = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.backbone2 = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        
        in_features = self.backbone1.num_features + self.backbone2.num_features
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features1 = self.backbone1(x)
        features2 = self.backbone2(x)
        features = torch.cat([features1, features2], dim=1)
        return self.head(features)

# ---------------------------------------------------------
# 1b. CONVNEXTV2-TINY MODEL (standalone) -> 9 classes
# ---------------------------------------------------------
class ConvNextModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 1c. EVA02-SMALL MODEL (standalone) -> 9 classes
# ---------------------------------------------------------
class Eva02Model(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = timm.create_model('eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 1d. SWINV2-SMALL MODEL (standalone) -> 9 classes
# ---------------------------------------------------------
class SwinV2Model(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = timm.create_model('swinv2_small_window8_256.ms_in1k', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 2. SEGMENTATION UNET MODEL (smp.Unet - mit_b5)
# ---------------------------------------------------------
# using segmentation_models_pytorch instead of manual Unet


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_LOADED = False
model_a = None
model_b = None
model_c = None
model_d = None

try:
    # 1. Load Model A: EnsembleModel (efficientnet_b3 + convnext_tiny) -> 8 classes
    DIAG_MODEL_PATH_A = "models/best_model_isic2019.pth"
    model_a = EnsembleModel(num_classes=8)
    ckpt_a = torch.load(DIAG_MODEL_PATH_A, map_location=device)
    sd_a = ckpt_a['state_dict'] if 'state_dict' in ckpt_a else ckpt_a
    model_a.load_state_dict(sd_a)
    model_a.to(device)
    model_a.eval()
    print(f"[OK] Loaded Model A (EnsembleModel, 8 classes) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Model A: {e}")
    model_a = None

try:
    # 2. Load Model B: ConvNeXtV2-Tiny -> 9 classes
    DIAG_MODEL_PATH_B = "models/best_model.pth"
    model_b = ConvNextModel(num_classes=9)
    ckpt_b = torch.load(DIAG_MODEL_PATH_B, map_location=device)
    sd_b = ckpt_b['state_dict'] if 'state_dict' in ckpt_b else ckpt_b
    # Strip 'model.' prefix if present (checkpoint was saved with a wrapper)
    sd_b = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd_b.items()}
    model_b.model.load_state_dict(sd_b)
    model_b.to(device)
    model_b.eval()
    print(f"[OK] Loaded Model B (ConvNeXtV2-Tiny, 9 classes) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Model B: {e}")
    model_b = None

try:
    # 3. Load Model C: Eva02-Small -> 9 classes
    DIAG_MODEL_PATH_C = "models/best_eva02_small_patch14_336.pth"
    model_c = Eva02Model(num_classes=9)
    ckpt_c = torch.load(DIAG_MODEL_PATH_C, map_location=device)
    sd_c = ckpt_c['state_dict'] if 'state_dict' in ckpt_c else ckpt_c
    # Strip 'model.' prefix if present
    sd_c = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd_c.items()}
    model_c.model.load_state_dict(sd_c)
    model_c.to(device)
    model_c.eval()
    print(f"[OK] Loaded Model C (Eva02-Small, 9 classes) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Model C: {e}")
    model_c = None

try:
    # 4. Load Model D: Swinv2-Small -> 9 classes
    DIAG_MODEL_PATH_D = "models/best_swinv2_small_window8_256.pth"
    model_d = SwinV2Model(num_classes=9)
    ckpt_d = torch.load(DIAG_MODEL_PATH_D, map_location=device)
    sd_d = ckpt_d['state_dict'] if 'state_dict' in ckpt_d else ckpt_d
    # Strip 'model.' prefix if present
    sd_d = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd_d.items()}
    model_d.model.load_state_dict(sd_d)
    model_d.to(device)
    model_d.eval()
    print(f"[OK] Loaded Model D (Swinv2-Small, 9 classes) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Model D: {e}")
    model_d = None

try:
    # 5. Load Segmentation Model (mit_b5 Unet)
    SEG_MODEL_PATH = "models/best_mit_b5_unet.pth"
    print("Loading segmentation model (339MB)...")
    model_seg = smp.Unet(
        encoder_name="mit_b5",
        encoder_weights=None, 
        in_channels=3,
        classes=1
    )
    # use map_location=device to load to GPU/CPU correctly
    model_seg.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
    model_seg.to(device)
    model_seg.eval()
    print(f"[OK] Loaded Segmentation Model (mit_b5) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Segmentation Model: {e}")

NUM_ENSEMBLE_CLASSES = 9  # max classes across all models

if any(m is not None for m in [model_a, model_b, model_c, model_d]):
    MODEL_LOADED = True
    loaded_count = sum(1 for m in [model_a, model_b, model_c, model_d] if m is not None)
    print(f"\n=== Ensemble ready with {loaded_count}/4 diagnosis model(s) ===")
else:
    MODEL_LOADED = False
    print("\n=== No diagnosis models loaded! ===")

LABELS = ["MEL (Melanoma)", "NV (Melanocytic nevus)", "BCC (Basal cell carcinoma)", 
          "AK (Actinic keratosis)", "BKL (Benign keratosis)", "DF (Dermatofibroma)", 
          "VASC (Vascular lesion)", "SCC (Squamous cell carcinoma)", "UNK (Unknown)"]

def calculate_asymmetry(mask):
    """Tính toán mức độ bất đối xứng của nốt ruồi."""
    if np.sum(mask) == 0: return 0
    
    # Tìm tâm của vật thể
    coords = np.column_stack(np.where(mask > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    roi = mask[y_min:y_max+1, x_min:x_max+1]
    h, w = roi.shape
    
    # Lật ảnh theo chiều dọc và ngang
    h_flip = cv2.flip(roi, 1)
    v_flip = cv2.flip(roi, 0)
    
    # So sánh sự khác biệt
    diff_h = np.logical_xor(roi, h_flip).sum()
    diff_v = np.logical_xor(roi, v_flip).sum()
    
    # Chuẩn hóa (0-100)
    total_pixels = roi.sum()
    score = ((diff_h + diff_v) / (2 * total_pixels)) * 100
    return min(100, max(0, float(score)))

def calculate_border(mask):
    """Tính toán độ gồ ghề của đường viền (Circularity)."""
    if np.sum(mask) == 0: return 0
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter == 0: return 0
    
    # Chỉ số Circularity: 4*pi*A / P^2
    # Hình tròn hoàn hảo có circularity = 1. Càng thấp càng gồ ghề.
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # Chuyển thành thang điểm rủi ro (circularity thấp -> điểm cao)
    score = (1.0 - circularity) * 100
    return min(100, max(0, float(score)))

def calculate_color(image_np, mask):
    """Phân tích sự đa dạng màu sắc bên trong vùng mask."""
    if np.sum(mask) == 0: return 0
    
    # Lấy các pixel bên trong mask
    pixels = image_np[mask > 0]
    
    # Tính độ lệch chuẩn của các kênh màu
    std_r = np.std(pixels[:, 0])
    std_g = np.std(pixels[:, 1])
    std_b = np.std(pixels[:, 2])
    
    # Trung bình độ lệch chuẩn
    avg_std = (std_r + std_g + std_b) / 3.0
    
    # Chuẩn hóa (giả định std > 50 là rất đa dạng màu sắc)
    score = (avg_std / 50.0) * 100
    return min(100, max(0, float(score)))


def extract_abcde_scores(abcde_dict):
    """
    Trích xuất điểm số A, B, C, D thô từ cấu trúc dữ liệu lưu trữ (Firestore/API).
    """
    if not abcde_dict:
        return 0.0, 0.0, 0.0, 0.0
        
    a = 0.0
    if "A_score" in abcde_dict:
        a = abcde_dict["A_score"]
    elif "A_asymmetry" in abcde_dict:
        a = abcde_dict["A_asymmetry"].get("score", 0.0)
        
    b = 0.0
    if "B_score" in abcde_dict:
        b = abcde_dict["B_score"]
    elif "B_border" in abcde_dict:
        b = abcde_dict["B_border"].get("score", 0.0)
        
    c = 0.0
    if "C_score" in abcde_dict:
        c = abcde_dict["C_score"]
    elif "C_color" in abcde_dict:
        c_val = abcde_dict["C_color"]
        if isinstance(c_val, dict):
            c = c_val.get("score", 0.0)
        else:
            c = float(c_val)
            
    d = 0.0
    if "D_score" in abcde_dict:
        d = abcde_dict["D_score"]
    elif "D_diameter" in abcde_dict:
        d_val = abcde_dict["D_diameter"]
        if isinstance(d_val, dict):
            d = d_val.get("score", 0.0)
        else:
            d = float(d_val)
            
    return float(a), float(b), float(c), float(d)


def calculate_evolution_score_and_status(curr_abcde_scores, prev_abcde_dict, curr_risk, prev_risk):
    """
    Tính toán chỉ số E (Evolution) và trạng thái biến đổi dựa trên so sánh giữa 2 lần quét.
    """
    try:
        prev_A, prev_B, prev_C, prev_D = extract_abcde_scores(prev_abcde_dict)
        
        curr_A = curr_abcde_scores.get("A", 0.0)
        curr_B = curr_abcde_scores.get("B", 0.0)
        curr_C = curr_abcde_scores.get("C", 0.0)
        curr_D = curr_abcde_scores.get("D", 0.0)
        
        delta_risk = float(curr_risk) - float(prev_risk)
        delta_D = curr_D - prev_D
        delta_A = curr_A - prev_A
        delta_B = curr_B - prev_B
        delta_C = curr_C - prev_C
        
        # Chỉ tính toán biến đổi xấu đi (tăng kích thước, bất đối xứng hơn, v.v.)
        pos_deltas = 0.0
        if delta_risk > 0:
            pos_deltas += delta_risk * 1.0
        if delta_D > 0:
            pos_deltas += delta_D * 1.5
        if delta_A > 0:
            pos_deltas += delta_A * 0.5
        if delta_B > 0:
            pos_deltas += delta_B * 0.5
        if delta_C > 0:
            pos_deltas += delta_C * 0.5
            
        if pos_deltas <= 3.0:
            e_score = max(0.0, min(100.0, 10.0 + delta_risk))
            e_status = "Stable"
        elif pos_deltas <= 15.0:
            e_score = 10.0 + pos_deltas * 2.0
            e_status = "Minor Evolution"
        else:
            e_score = min(100.0, 40.0 + (pos_deltas - 15.0) * 1.5)
            e_status = "Significant Evolution"
            
        return round(float(e_score), 1), e_status
    except Exception as e:
        print(f"[ERROR] Failed to calculate evolution: {e}")
        return 0.0, "Requires History"


def process_image(image_bytes: bytes):
    if not MODEL_LOADED:
        return None
        
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = original_image.size
    
    # ----------------------------------
    # 1. SEGMENTATION PIPELINE (mit_b5_unet)
    # ----------------------------------
    # Convert PIL Image to numpy (RGB format)
    ori_img_np = np.array(original_image)
    
    # Image size specific to the new model
    img_size = 384
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])
    
    input_tensor = transform(image=ori_img_np)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model_seg(input_tensor)
        mask_pred = torch.sigmoid(output).cpu().numpy()[0][0]
        
    # Resize mask back to original image size
    mask_full = cv2.resize(mask_pred, (W, H))
    binary_mask = (mask_full > 0.5).astype(np.uint8)
    
    # Auto-Crop Logic
    coords = np.column_stack(np.where(binary_mask > 0))
    padding = 0.2
    
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        h_m, w_m = y_max - y_min, x_max - x_min
        rmin = max(0, int(y_min - h_m * padding))
        rmax = min(H, int(y_max + h_m * padding))
        cmin = max(0, int(x_min - w_m * padding))
        cmax = min(W, int(x_max + w_m * padding))
    else:
        rmin, rmax, cmin, cmax = 0, H, 0, W
        
    # Crop Image (PIL)
    cropped_image = original_image.crop((cmin, rmin, cmax, rmax))
    
    # Calculate bounding box for Frontend (values from 0 to 1 relative to original image)
    bbox = {
        "x": cmin / W,
        "y": rmin / H,
        "w": (cmax - cmin) / W,
        "h": (rmax - rmin) / H
    }
    
    # Calculate ABCDE features from segmentation
    a_score = calculate_asymmetry(binary_mask)
    b_score = calculate_border(binary_mask)
    c_score = calculate_color(ori_img_np, binary_mask)
    # D (Diameter): relative to image size, simplified
    d_score = min(100, ((cmax - cmin) * (rmax - rmin) / (W * H)) * 500) 
    
    abcde_results = {
        "A": a_score,
        "B": b_score,
        "C": c_score,
        "D": d_score,
        "E": 0.0 # Placeholder for single scan
    }
    
    # ----------------------------------
    # 2. DIAGNOSIS PIPELINE - WEIGHTED ENSEMBLE 2 MODEL + TTA x6 (On Cropped)
    #    Kỹ thuật: Weighted Average Probabilities + AMP fp16
    # ----------------------------------
    
    # 6 TTA augmentations (giữ nguyên tâm ảnh, không crop)
    tta_augments = [
        lambda img: img,                                          # 1. Original
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),         # 2. Horizontal flip
        lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),         # 3. Vertical flip
        lambda img: img.transpose(Image.ROTATE_90),               # 4. Rotate 90°
        lambda img: img.transpose(Image.ROTATE_270),              # 5. Rotate -90° (270°)
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),  # 6. HFlip+VFlip (≡ Rotate 180°)
    ]
    
    diag_transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    diag_transform_256 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    diag_transform_336 = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    with torch.no_grad():
        probs_list_a = []  # Model A
        probs_list_b = []  # Model B
        probs_list_c = []  # Model C (Eva02)
        probs_list_d = []  # Model D (SwinV2)
        
        for tta_fn in tta_augments:
            aug_image = tta_fn(cropped_image)
            tensor_224 = diag_transform_224(aug_image).unsqueeze(0).to(device)
            tensor_256 = diag_transform_256(aug_image).unsqueeze(0).to(device)
            tensor_336 = diag_transform_336(aug_image).unsqueeze(0).to(device)
            
            # === Mixed Precision (AMP fp16) ===
            with torch.amp.autocast(device_type=device_type):
                # Model A: EnsembleModel (8 classes -> pad to 9)
                if model_a is not None:
                    outputs_a = model_a(tensor_224)
                    probs_a = torch.nn.functional.softmax(outputs_a[0], dim=0)
                    probs_a = torch.cat([probs_a, torch.zeros(NUM_ENSEMBLE_CLASSES - 8, device=probs_a.device)])
                    probs_list_a.append(probs_a)
                
                # Model B: ConvNeXtV2-Tiny (9 classes) 
                if model_b is not None:
                    outputs_b = model_b(tensor_224)
                    probs_b = torch.nn.functional.softmax(outputs_b[0], dim=0)
                    probs_list_b.append(probs_b)

                # Model C: Eva02-Small (9 classes)
                if model_c is not None:
                    outputs_c = model_c(tensor_336)
                    probs_c = torch.nn.functional.softmax(outputs_c[0], dim=0)
                    probs_list_c.append(probs_c)

                # Model D: Swinv2-Small (9 classes)
                if model_d is not None:
                    outputs_d = model_d(tensor_256)
                    probs_d = torch.nn.functional.softmax(outputs_d[0], dim=0)
                    probs_list_d.append(probs_d)
        
        # Average TTA for each model
        avg_probs_list = []
        if probs_list_a:
            avg_probs_a = torch.stack(probs_list_a).mean(dim=0)
            avg_probs_list.append(avg_probs_a)
            pred_idx = torch.argmax(avg_probs_a).item()
            lbl = LABELS[pred_idx] if pred_idx < len(LABELS) else "Unknown"
            print(f"[Ensemble Log] Model A (EfficientNet+ConvNeXt) -> {lbl}: {avg_probs_a[pred_idx].item()*100:.2f}%")
        if probs_list_b:
            avg_probs_b = torch.stack(probs_list_b).mean(dim=0)
            avg_probs_list.append(avg_probs_b)
            pred_idx = torch.argmax(avg_probs_b).item()
            lbl = LABELS[pred_idx] if pred_idx < len(LABELS) else "Unknown"
            print(f"[Ensemble Log] Model B (ConvNeXtV2) -> {lbl}: {avg_probs_b[pred_idx].item()*100:.2f}%")
        if probs_list_c:
            avg_probs_c = torch.stack(probs_list_c).mean(dim=0)
            avg_probs_list.append(avg_probs_c)
            pred_idx = torch.argmax(avg_probs_c).item()
            lbl = LABELS[pred_idx] if pred_idx < len(LABELS) else "Unknown"
            print(f"[Ensemble Log] Model C (Eva02) -> {lbl}: {avg_probs_c[pred_idx].item()*100:.2f}%")
        if probs_list_d:
            avg_probs_d = torch.stack(probs_list_d).mean(dim=0)
            avg_probs_list.append(avg_probs_d)
            pred_idx = torch.argmax(avg_probs_d).item()
            lbl = LABELS[pred_idx] if pred_idx < len(LABELS) else "Unknown"
            print(f"[Ensemble Log] Model D (SwinV2) -> {lbl}: {avg_probs_d[pred_idx].item()*100:.2f}%")
        
        # === Dynamic Average Ensemble ===
        if avg_probs_list:
            final_probs = torch.stack(avg_probs_list).mean(dim=0)
            ensemble_idx = torch.argmax(final_probs).item()
            ensemble_lbl = LABELS[ensemble_idx] if ensemble_idx < len(LABELS) else "Unknown"
            print(f"[Ensemble Log] FINAL ENSEMBLE -> {ensemble_lbl}: {final_probs[ensemble_idx].item()*100:.2f}%")
        else:
            # Fallback: no models loaded
            return None
        
        confidence, predicted = torch.max(final_probs, 0)
        
        # Calculate Top 3 predictions
        top3_prob, top3_indices = torch.topk(final_probs, 3)
        top3 = []
        for i in range(3):
            idx = top3_indices[i].item()
            label_name = LABELS[idx] if idx < len(LABELS) else "Unknown"
            score = round(float(top3_prob[i].item()) * 100, 2)
            top3.append({"label": label_name, "score": score})
            
    # --- Generate Heatmap Overlay using mit_b5_unet segmentation ---
    try:
        mask_255 = (mask_full * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_255, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(ori_img_np, 0.65, heatmap_rgb, 0.35, 0)
        overlay_pil = Image.fromarray(overlay)
    except Exception as e:
        print(f"[WARN] Failed to generate heatmap overlay: {e}")
        overlay_pil = original_image

    return int(predicted.item()), float(confidence.item()), bbox, top3, abcde_results, overlay_pil


def get_current_user(authorization: str = Form(None)):
    """Helper to verify Firebase token from Form data or Header."""
    if not authorization:
        return None
    
    token = authorization.replace("Bearer ", "")
    user = firebase_auth_manager.verify_id_token(token)
    return user

@app.on_event("startup")
def startup_event():
    # Run once initially in background
    threading.Thread(target=run_explore_workflow, daemon=True).start()
    
    # Schedule to run every 1 hour
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_explore_workflow, 'interval', hours=1)
    scheduler.start()
    print("[Scheduler] LangGraph Explore Agent started.")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "SkinGuardAI API is running"}

@app.get("/api/explore")
async def get_explore_content(language: str = "vi"):
    """Fetch cached LangGraph blog posts."""
    blogs = get_latest_blogs()
    if not blogs:
        # Fallback if somehow empty or still loading
        blogs = [
            { "category": "Khởi tạo", "title": "Hệ thống đang phân tích xu hướng", "desc": "Các bài viết mới đang được AI tổng hợp bằng LangGraph. Vui lòng tải lại trang sau ít phút.", "img": "https://images.unsplash.com/photo-1551076805-e1869033e561" }
        ]
    return {"status": "success", "data": blogs}

@app.post("/api/history")
async def get_user_history(authorization: str = Form(...)):
    user = get_current_user(authorization)
    if not user:
        return {"status": "error", "message": "Unauthorized"}, 401
    
    user_id = user['uid']
    try:
        db = firebase_auth_manager.get_db()
        if not db:
            return {"status": "error", "message": "Database not initialized"}, 500

        docs = db.collection("analysis_records")\
                 .where("user_id", "==", user_id)\
                 .get()
        
        data = []
        for doc in docs:
            item = doc.to_dict()
            item['id'] = doc.id
            data.append(item)
            
        # Sắp xếp theo ngày tạo mới nhất lên đầu
        data.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
        return {"status": "success", "data": data}
    except Exception as e:
        print(f"[ERROR] Failed to fetch history from Firestore: {e}")
        return {"status": "error", "message": str(e)}, 500

class AnalyzeResponse(BaseModel):
    risk_score: float
    classification: str
    confidence: float
    abcde: dict
    bbox: dict
    top3: list
    image_url: str = None
    heatmap_url: str = None
    medical_advice: str = None
    uv_index: float = None
    temperature: float = None
    location: str = None

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    request: Request,
    file: UploadFile = File(...), 
    authorization: str = Form(None),
    uv_index: float = Form(None),
    temperature: float = Form(None),
    location: str = Form(None),
    prev_abcde: str = Form(None),
    prev_risk: float = Form(None)
):
    contents = await file.read()
    
    # Run Pipeline
    result = process_image(contents)
    
    e_score = 0.0
    e_status = "Requires History"
    
    if result:
        class_idx, confidence, bbox, top3, abcde_scores, overlay_pil = result
        classification = LABELS[class_idx] if class_idx < len(LABELS) else "Unknown"
        
        if "MEL" in classification or "BCC" in classification or "SCC" in classification:
            base_risk = round(confidence * 100, 1)
        else:
            base_risk = round((1.0 - confidence) * 100, 1)
            
        # --- Compute E (Evolution) dynamic score and status ---
        prev_abcde_dict = None
        prev_risk_val = None
        
        if prev_abcde:
            try:
                prev_abcde_dict = json.loads(prev_abcde)
            except Exception:
                pass
        if prev_risk is not None:
            prev_risk_val = prev_risk

        user = get_current_user(authorization)
        if not prev_abcde_dict and user:
            try:
                user_id = user['uid']
                db = firebase_auth_manager.get_db()
                if db:
                    docs = db.collection("analysis_records")\
                             .where("user_id", "==", user_id)\
                             .order_by("created_at", direction="DESCENDING")\
                             .limit(1)\
                             .get()
                    if docs and len(docs) > 0:
                        prev_record = docs[0].to_dict()
                        prev_abcde_dict = prev_record.get("abcde", {})
                        prev_risk_val = prev_record.get("risk_score", 0.0)
            except Exception as e:
                print(f"[WARN] Failed to fetch previous scan for evolution: {e}")

        if prev_abcde_dict:
            e_score, e_status = calculate_evolution_score_and_status(
                abcde_scores, prev_abcde_dict, base_risk, prev_risk_val or 0.0
            )

        # Real ABCDE logic from calculation
        abcde_final = {
            "A_score": abcde_scores["A"],
            "B_score": abcde_scores["B"],
            "C_score": abcde_scores["C"],
            "D_score": abcde_scores["D"],
            "E_score": e_score
        }
    else:
        # Fallback if processing failed
        return {"status": "error", "message": "Model processing failed"}, 500

    # --- Save Image and Heatmap Overlay to ImageKit Cloud ---
    try:
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".jpg"
        unique_id = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        
        local_filename = f"{timestamp}_{unique_id}{ext}"
        heatmap_filename = f"heatmap_{timestamp}_{unique_id}{ext}"
        
        # Chuyển đổi overlay PIL sang bytes
        img_byte_arr = io.BytesIO()
        overlay_pil.save(img_byte_arr, format='JPEG')
        overlay_bytes = img_byte_arr.getvalue()
        
        image_url = upload_to_imagekit(contents, local_filename, folder="/scans")
        heatmap_url = upload_to_imagekit(overlay_bytes, heatmap_filename, folder="/heatmaps")
        print(f"[OK] Uploaded to ImageKit: {image_url}")
    except Exception as e:
        print(f"[ERROR] Failed to upload images to ImageKit: {e}")
        image_url = ""
        heatmap_url = ""

    # --- Save to Firestore Database if User is Auth ---
    user = get_current_user(authorization)
    if user and image_url:
        try:
            user_id = user['uid']
            db = firebase_auth_manager.get_db()
            if db:
                record_id = str(uuid.uuid4())
                record_data = {
                    "id": record_id,
                    "user_id": user_id,
                    "image_url": image_url,
                    "heatmap_url": heatmap_url,
                    "risk_score": base_risk,
                    "classification": classification,
                    "confidence": round(confidence * 100 if result else confidence, 2),
                    "abcde": abcde_final,
                    "top3": top3,
                    "uv_index": uv_index,
                    "temperature": temperature,
                    "location": location,
                    "created_at": datetime.utcnow().isoformat()
                }
                db.collection("analysis_records").document(record_id).set(record_data)
                print(f"[OK] Scan saved to Firestore with ID: {record_id}")
        except Exception as e:
            print(f"[ERROR] Failed to save to Firestore database: {e}")
    
    # --- Generate Medical Advice using AI ---
    advice_prompt = (
        f"Phân tích kết quả scan da: Chẩn đoán là {classification} với độ tin cậy {confidence*100:.1f}%. "
        f"Mức độ rủi ro tổng thể: {base_risk}%. "
        "Hãy đưa ra lời khuyên y khoa ngắn gọn (2-3 câu), chuyên nghiệp nhưng dễ hiểu cho bệnh nhân. "
        "Nhấn mạnh vào việc theo dõi hoặc đi khám nếu cần thiết. Trả lời bằng tiếng Việt."
    )
    
    medical_advice = "Hệ thống đang chuẩn bị lời khuyên..."
    try:
        advice_res = nv_client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
            messages=[{"role": "user", "content": advice_prompt}],
            max_tokens=256,
            temperature=0.3,
            stream=False
        )
        medical_advice = advice_res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] Failed to generate AI advice: {e}")
        medical_advice = "Cảnh báo: AI phát hiện dấu hiệu bất thường. Hãy đặt lịch hẹn với bác sĩ da liễu để kiểm tra chi tiết." if base_risk > 50 else "Kết quả cho thấy nguy cơ thấp. Hãy tiếp tục theo dõi định kỳ."

    return {
        "risk_score": base_risk,
        "classification": classification,
        "confidence": round(confidence * 100 if result else confidence, 2),
        "medical_advice": medical_advice,
        "abcde": {
            "A_asymmetry": {
                "score": abcde_final["A_score"],
                "status": "High" if abcde_final["A_score"] > 50 else "Low"
            },
            "B_border": {
                "score": abcde_final["B_score"],
                "status": "Abnormal" if abcde_final["B_score"] > 50 else "Normal"
            },
            "C_color": {
                "score": abcde_final["C_score"],
                "status": "Moderate" if abcde_final["C_score"] > 50 else "Uniform"
            },
            "D_diameter": {
                "value": round(abcde_final["D_score"] / 10.0, 1),
                "score": abcde_final["D_score"]
            },
            "E_evolution": {
                "score": abcde_final["E_score"],
                "status": e_status
            }
        },
        "bbox": bbox,
        "top3": top3,
        "image_url": image_url,
        "heatmap_url": heatmap_url,
        "uv_index": uv_index,
        "temperature": temperature,
        "location": location
    }


@app.post("/api/upload-avatar")
async def upload_avatar(request: Request, file: UploadFile = File(...), authorization: str = Form(None)):
    """Upload user avatar to ImageKit Cloud Storage."""
    user = get_current_user(authorization)
    if not user:
        return {"status": "error", "message": "Unauthorized"}, 401
    
    try:
        contents = await file.read()
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".jpg"
        
        filename = f"avatar_{user['uid']}{ext}"
        image_url = upload_to_imagekit(contents, filename, folder="/avatars")
        
        if not image_url:
            return {"status": "error", "message": "Failed to upload avatar to ImageKit"}, 500

        return {"status": "success", "url": image_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


class CompareRequest(BaseModel):
    scan1: dict
    scan2: dict
    language: str = "vi"

@app.post("/api/compare")
async def compare_scans(req: CompareRequest):
    s1 = req.scan1
    s2 = req.scan2
    
    # Extract values safely
    classification1 = s1.get("classification", s1.get("diagnosis", "Chưa rõ"))
    classification2 = s2.get("classification", s2.get("diagnosis", "Chưa rõ"))
    
    confidence1 = s1.get("confidence", 0)
    confidence2 = s2.get("confidence", 0)
    
    risk1 = s1.get("risk_score", s1.get("risk", 0))
    risk2 = s2.get("risk_score", s2.get("risk", 0))
    
    date1 = s1.get("created_at", s1.get("date", "Lần quét 1"))
    date2 = s2.get("created_at", s2.get("date", "Lần quét 2"))
    
    abcde1 = s1.get("abcde") or {}
    abcde2 = s2.get("abcde") or {}
    
    # Safe ABCDE extraction
    def get_abcde_summary(abcde):
        if not abcde:
            return "Không có dữ liệu chi tiết."
        
        # Check if shape is { A_asymmetry: { score: X, status: Y } } or raw { A_score: X }
        if "A_asymmetry" in abcde:
            a = f"{abcde.get('A_asymmetry', {}).get('score', 0)}% ({abcde.get('A_asymmetry', {}).get('status', 'N/A')})"
            b = f"{abcde.get('B_border', {}).get('score', 0)}% ({abcde.get('B_border', {}).get('status', 'N/A')})"
            c = f"{abcde.get('C_color', {}).get('score', 0)}% ({abcde.get('C_color', {}).get('status', 'N/A')})"
            d = f"{abcde.get('D_diameter', {}).get('value', 0.0)}mm (Điểm: {abcde.get('D_diameter', {}).get('score', 0)}%)"
            e = f"{abcde.get('E_evolution', {}).get('score', 0)}% ({abcde.get('E_evolution', {}).get('status', 'N/A')})"
        else:
            a = f"{abcde.get('A_score', 0)}%"
            b = f"{abcde.get('B_score', 0)}%"
            c = f"{abcde.get('C_color', 0)}%" if "C_color" in abcde else f"{abcde.get('C_score', 0)}%"
            d = f"{abcde.get('D_diameter', 0)}%" if "D_diameter" in abcde else f"{abcde.get('D_score', 0)}%"
            e = f"{abcde.get('E_score', 0)}%"
        return f"A_Asymmetry: {a}, B_Border: {b}, C_Color: {c}, D_Diameter: {d}, E_Evolution: {e}"

    summary1 = get_abcde_summary(abcde1)
    summary2 = get_abcde_summary(abcde2)
    
    uv1 = s1.get("uv_index", "N/A")
    uv2 = s2.get("uv_index", "N/A")
    
    temp1 = s1.get("temperature", "N/A")
    temp2 = s2.get("temperature", "N/A")
    
    loc1 = s1.get("location", "N/A")
    loc2 = s2.get("location", "N/A")
    
    prompt = (
        "Bạn là Skinderm AI, một chuyên gia AI da liễu hàng đầu được tích hợp trong hệ thống chẩn đoán. "
        "Nhiệm vụ của bạn là phân tích so sánh tiến triển của hai lượt quét da (scan) khác nhau của bệnh nhân để giúp bác sĩ lâm sàng và bệnh nhân hiểu rõ tình trạng thay đổi.\n\n"
        "=== THÔNG TIN LƯỢT QUÉT 1 (CŨ HƠN) ===\n"
        f"- Ngày quét: {date1}\n"
        f"- Chẩn đoán chính: {classification1} (Độ tin cậy: {confidence1}%)\n"
        f"- Mức độ rủi ro tổng hợp: {risk1}%\n"
        f"- Chỉ số chi tiết ABCDE: {summary1}\n"
        f"- Môi trường thời tiết: Vị trí {loc1}, Chỉ số UV {uv1}, Nhiệt độ {temp1}°C\n\n"
        
        "=== THÔNG TIN LƯỢT QUÉT 2 (MỚI HƠN) ===\n"
        f"- Ngày quét: {date2}\n"
        f"- Chẩn đoán chính: {classification2} (Độ tin cậy: {confidence2}%)\n"
        f"- Mức độ rủi ro tổng hợp: {risk2}%\n"
        f"- Chỉ số chi tiết ABCDE: {summary2}\n"
        f"- Môi trường thời tiết: Vị trí {loc2}, Chỉ số UV {uv2}, Nhiệt độ {temp2}°C\n\n"
        
        "Nhiệm vụ của bạn: Hãy phân tích tiến triển và viết một Báo cáo So sánh Tiến triển Y khoa cực kỳ chi tiết, khoa học bằng tiếng Việt. Hãy định dạng báo cáo thật đẹp bằng Markdown:\n"
        "1. Sử dụng các tiêu đề rõ ràng (### 1. Phân tích Xu hướng Rủi ro, ### 2. Đánh giá Biến động Chỉ số ABCDE, ### 3. Đánh giá Môi trường & Lối sống, ### 4. Khuyến nghị Y khoa chuyên sâu).\n"
        "2. Sử dụng bảng Markdown (Markdown Table) để hiển thị so sánh đối chiếu giữa Lần 1 và Lần 2 (so sánh: Chẩn đoán, Rủi ro, Kích thước D, Chỉ số Asymmetry, UV Index).\n"
        "3. Dùng khối trích dẫn (Blockquote) màu đỏ nhạt để đưa ra các lời khuyên an toàn y tế khẩn cấp nếu phát hiện rủi ro tăng lên hoặc kích thước nốt ruồi to ra.\n"
        "4. Nhấn mạnh (Bôi đậm) các biến động đáng chú ý.\n\n"
        "Các ranh giới đỏ cấm kỵ (Hard Guardrails):\n"
        "- KHÔNG BAO GIỜ khẳng định bệnh nhân bị ung thư da hay tự ý chẩn đoán xác định.\n"
        "- Tuyệt đối không kê đơn thuốc hoặc chỉ định can thiệp phẫu thuật trực tiếp.\n\n"
        "Hãy luôn chèn câu miễn trừ trách nhiệm y tế chuẩn mực ở cuối báo cáo: 'Lưu ý: Báo cáo so sánh này được thực hiện tự động bởi Skinderm AI và chỉ mang tính chất sàng lọc tham khảo, không thay thế chẩn đoán lâm sàng của bác sĩ chuyên khoa da liễu.'"
    )
    
    try:
        completion = nv_client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3
        )
        report = completion.choices[0].message.content.strip()
        return {"status": "success", "report": report}
    except Exception as e:
        print(f"[ERROR] LLM Compare error: {e}")
        diff_risk = risk2 - risk1
        risk_dir = "tăng lên" if diff_risk > 0 else ("giảm xuống" if diff_risk < 0 else "ổn định")
        fallback_report = (
            f"### Báo cáo So sánh Tự động (Hệ thống dự phòng)\n\n"
            f"- **Xu hướng Rủi ro**: Mức độ rủi ro đã biến động {risk_dir} từ **{risk1}%** ở lần quét thứ nhất sang **{risk2}%** ở lần quét thứ hai (lệch {abs(diff_risk)}%).\n"
            f"- **Chẩn đoán**: Từ **{classification1}** sang **{classification2}**.\n\n"
            f"> **KHUYẾN NGHỊ:** Lịch sử ghi nhận tiến triển có biến động. Xin vui lòng đặt lịch khám sớm với bác sĩ chuyên khoa da liễu tại cơ sở y tế gần nhất để đối chiếu lâm sàng cụ thể."
        )
        return {"status": "success", "report": fallback_report}


class ChatRequest(BaseModel):
    message: str
    language: str = "vi"
    model: str = "meta/llama-3.2-90b-vision-instruct"
    context: str = ""

@app.post("/api/chat")
async def chat_interaction(chat_req: ChatRequest, request: Request):
    system_prompt = (
        "Bạn là Skinderm AI, một trợ lý trí tuệ nhân tạo chuyên cung cấp thông tin hỗ trợ về các vấn đề da liễu. "
        "Bạn KHÔNG PHẢI là bác sĩ y khoa, không có chứng chỉ hành nghề, và bạn phải luôn tự nhận thức rõ vai trò là một máy móc hỗ trợ, tuyệt đối không được nhận mình là bác sĩ con người.\n"
        "Hãy trả lời bằng tiếng Việt, với phong cách chuyên nghiệp, thân thiện, tận tâm, gần gũi và ấm áp, tránh sự khô khan hay cứng ngắc. ĐẶC BIỆT: Hãy giải thích mọi thứ bằng từ ngữ dễ hiểu nhất, tránh các thuật ngữ chuyên ngành phức tạp, hoặc nếu bắt buộc phải dùng thì phải định nghĩa rõ ràng để một người bình thường không biết gì về y tế cũng hiểu được.\n\n"
        "Dựa trên dữ liệu chẩn đoán sau: " + chat_req.context + "\n\n"
        "Nhiệm vụ của bạn là giải thích kết quả phân loại hình ảnh tổn thương da, cung cấp các thông tin y khoa cơ bản, nguyên nhân, và cách chăm sóc da thông thường liên quan đến 9 loại bệnh da liễu trong hệ thống. Tuyệt đối không trả lời các câu hỏi ngoài lề không liên quan đến da liễu.\n\n"
        "Hãy cấu trúc câu trả lời của bạn thật chuyên nghiệp và khoa học bằng cách sử dụng Markdown:\n"
        "- Sử dụng các tiêu đề mục rõ ràng (như ### 1. Phân tích Chẩn đoán & Độ rủi ro, ### 2. Đánh giá Chỉ số ABCDE & Thời tiết, ### 3. Khuyến nghị Chăm sóc & Theo dõi).\n"
        "- Trình bày các dữ liệu so sánh hoặc chỉ số bằng bảng biểu (Markdown Table) để trực quan và dễ hiểu.\n"
        "- Dùng danh sách có dấu đầu dòng (Bullet points) ngắn gọn, súc tích.\n"
        "- Bôi đậm (Strong) các từ khóa y khoa quan trọng để nhấn mạnh.\n"
        "- Đặt các cảnh báo quan trọng hoặc lời khuyên khẩn cấp trong khối trích dẫn (Blockquote) để tạo sự chú ý đặc biệt.\n\n"
        "Các ranh giới đỏ cấm kỵ (Hard Guardrails):\n"
        "- TUYỆT ĐỐI KHÔNG TRẢ LỜI CÂU HỎI NGOÀI LỀ: Nếu người dùng hỏi bất kỳ câu hỏi nào KHÔNG liên quan trực tiếp đến da liễu, bệnh lý da, cách chăm sóc da, hoặc kết quả scan da (ví dụ: các thắc mắc về lập trình phần mềm, toán học, lịch sử, ẩm thực, làm thơ, hoặc các chuyên khoa y khoa hoàn toàn khác như tim mạch, cơ xương khớp, nha khoa...), bạn BẮT BUỘC phải từ chối trả lời một cách lịch sự, nhã nhặn. Hãy nêu rõ rằng bạn là AI chuyên biệt về Da liễu của Skinderm và chỉ có thể tư vấn các vấn đề trong phạm vi này.\n"
        "- KHÔNG BAO GIỜ được phép chẩn đoán khẳng định bệnh nhân mắc bệnh gì.\n"
        "- TUYỆT ĐỐI CẤM kê đơn thuốc, gợi ý liều lượng thuốc, hay chỉ định các biện pháp can thiệp y khoa (phẫu thuật, xạ trị,...).\n"
        "- Nếu người dùng yêu cầu kê đơn thuốc, hãy từ chối ngay lập tức một cách lịch sự, nhẹ nhàng và giải thích rõ giới hạn của mình.\n\n"
        "Luôn chèn câu này vào cuối các tư vấn về bệnh lý: 'Lưu ý: Skinderm AI chỉ mang tính chất hỗ trợ tầm soát. Vui lòng đến bệnh viện hoặc phòng khám da liễu để được bác sĩ chuyên khoa thăm khám và điều trị chính xác nhất.'"
    )

    async def generate_events():
        try:
            target_model = chat_req.model
            if "llama-3.1-8b" in target_model:
                target_model = "meta/llama-3.2-11b-vision-instruct"
            elif "llama-3.1-70b" in target_model:
                target_model = "meta/llama-3.2-90b-vision-instruct"

            completion = nv_client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chat_req.message}
                ],
                temperature=0.4,
                top_p=0.9,
                max_tokens=4096, # Set to reasonable limit
                stream=True,
                timeout=180.0
            )

            for chunk in completion:
                # Check if client is still there
                if await request.is_disconnected():
                    print("[INFO] Client disconnected. Stopping generation.")
                    break

                try:
                    if not getattr(chunk, "choices", None): continue
                    content = chunk.choices[0].delta.content
                    if content:
                        data = {"choices": [{"delta": {"content": content}}]}
                        yield f"data: {json.dumps(data)}\n\n"
                except Exception:
                    break

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    return StreamingResponse(generate_events(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
