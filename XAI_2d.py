# ==========================================
# 0. NNUNET ENVIRONMENT SETUP 
# ==========================================
import os

os.environ["nnUNet_raw"] = r"C:\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"C:\nnUNet_results"

# ==========================================
# IMPORTS
# ==========================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import blosc2
import torch.nn.functional as F
import math
from tqdm import tqdm
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from torch.cuda.amp import autocast

# ==========================================
# 1. USER SETTINGS
# ==========================================
CASES_TO_PROCESS = [
    "case_38.b2nd"
]

BASE_DATA_PATH = "/path/to/preprocessed_data"
MODEL_FOLDER = "/path/to/model_folder"
output_folder = "/path/to/output_directory"

os.makedirs(output_folder, exist_ok=True)

TARGET_CLASS = 3
ZOOM_FACTOR = 0.4 

# ==========================================
# 2. LOAD MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Loading Model on {device}...")

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=False,
    use_mirroring=False,
    device=device
)

predictor.initialize_from_trained_model_folder(
    model_training_output_dir=MODEL_FOLDER,
    use_folds=(0,),
    checkpoint_name="checkpoint_best.pth"
)

predictor.network.to(device)
model = predictor.network

# ==========================================
# 3. HELPER
# ==========================================
def next_power_of_2(n):
    return 2 ** math.ceil(math.log2(n))

# ==========================================
# 4. MAIN LOOP
# ==========================================
for case_filename in CASES_TO_PROCESS:
    print(f"\n========================================")
    print(f" Processing: {case_filename}")
    print(f"========================================")

    full_path = os.path.join(BASE_DATA_PATH, case_filename)
    vol = blosc2.open(full_path)[:].astype(np.float32)

    original_h, original_w = vol.shape[-2:]

    target_h = next_power_of_2(original_h)
    target_w = next_power_of_2(original_w)
    pad_h = target_h - original_h
    pad_w = target_w - original_w

    # ==========================================
    # AUTO SLICE FINDER
    # ==========================================
    if vol.ndim == 4:
        print(f" Scanning {vol.shape[1]} slices...")
        best_slice = 0
        max_score = -1

        for z in range(vol.shape[1]):
            x_test = torch.from_numpy(vol[:, z]).float().unsqueeze(0).to(device)
            x_test_pad = F.pad(x_test, (0, pad_w, 0, pad_h), mode='constant', value=0)

            with torch.no_grad():
                with autocast():
                    out = model(x_test_pad)
                    if isinstance(out, (list, tuple)):
                        out = out[0]

                    score = torch.softmax(out, dim=1)[0, TARGET_CLASS].sum().item()

                    if score > max_score:
                        max_score = score
                        best_slice = z

        x2d = vol[:, best_slice]
        print(f" Using slice {best_slice}")
    else:
        x2d = vol

    x = torch.from_numpy(x2d).float().unsqueeze(0).to(device)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
    x_padded.requires_grad_(True)

    ct_image = x[0, 0].detach().cpu().numpy()

    # ==========================================
    # GRAD-CAM
    # ==========================================
    print(" Grad-CAM...")

    activations, gradients = None, None

    def fwd_hook(m, i, o):
        global activations
        activations = o

    def bwd_hook(m, gi, go):
        global gradients
        gradients = go[0]

    target_layer = model.decoder.stages[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()

    out = model(x_padded)
    if isinstance(out, (list, tuple)):
        out = out[0]

    score = out[:, TARGET_CLASS].mean()
    score.backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1)).squeeze().detach().cpu().numpy()

    h1.remove()
    h2.remove()
    model.zero_grad()

    cam_crop = cam[:original_h, :original_w]
    cam_norm = (cam_crop - cam_crop.min()) / (cam_crop.max() - cam_crop.min() + 1e-8)

    # ==========================================
    # UNCERTAINTY
    # ==========================================
    print(" Uncertainty...")

    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.apply(enable_dropout)

    preds = []
    for _ in range(20):
        with torch.no_grad():
            out_mc = model(x_padded)
            if isinstance(out_mc, (list, tuple)):
                out_mc = out_mc[0]
            preds.append(torch.softmax(out_mc, dim=1).cpu().numpy())

    variance = np.var(np.stack(preds), axis=0)[0, TARGET_CLASS]
    unc_crop = variance[:original_h, :original_w]
    unc_norm = (np.power(unc_crop, 0.5) - unc_crop.min()) / (unc_crop.max() - unc_crop.min() + 1e-8)

    model.eval()

    # ==========================================
    # OCCLUSION
    # ==========================================
    print(" Occlusion...")

    PATCH_SIZE = 64
    STRIDE = 64

    heatmap_occ = np.zeros((target_h, target_w))
    counts = np.zeros((target_h, target_w))

    with torch.no_grad():
        base_out = model(x_padded)
        if isinstance(base_out, (list, tuple)):
            base_out = base_out[0]
        base_score = torch.softmax(base_out, dim=1)[0, TARGET_CLASS].sum().item()

    for y in range(0, target_h - PATCH_SIZE + 1, STRIDE):
        for x_c in range(0, target_w - PATCH_SIZE + 1, STRIDE):
            img_occ = x_padded.clone()
            img_occ[:, :, y:y+PATCH_SIZE, x_c:x_c+PATCH_SIZE] = x_padded.min()

            out_occ = model(img_occ)
            if isinstance(out_occ, (list, tuple)):
                out_occ = out_occ[0]

            new_score = torch.softmax(out_occ, dim=1)[0, TARGET_CLASS].sum().item()
            drop = base_score - new_score

            heatmap_occ[y:y+PATCH_SIZE, x_c:x_c+PATCH_SIZE] += drop
            counts[y:y+PATCH_SIZE, x_c:x_c+PATCH_SIZE] += 1

    occ_crop = (heatmap_occ / (counts + 1e-8))[:original_h, :original_w]
    occ_norm = (occ_crop - occ_crop.min()) / (occ_crop.max() - occ_crop.min() + 1e-8)

    # ==========================================
    # VISUALIZATION 
    # ==========================================
    plt.figure(figsize=(20, 6))

    y_center, x_center = original_h // 2, original_w // 2
    y_size, x_size = int(original_h * ZOOM_FACTOR), int(original_w * ZOOM_FACTOR)

    y1, y2 = y_center - y_size // 2, y_center + y_size // 2
    x1, x2 = x_center - x_size // 2, x_center + x_size // 2

    maps = [None, cam_norm, unc_norm, occ_norm]
    titles = ["A. Original CT", "B. Grad-CAM (Attention)", "C. Uncertainty (Confidence)", "D. Occlusion (Importance)"]

    for i in range(4):
        ax = plt.subplot(1, 4, i+1)
        plt.imshow(ct_image, cmap="gray")

        if maps[i] is not None:
            plt.imshow(maps[i], cmap="jet", alpha=0.5)

        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)

        plt.title(titles[i])
        plt.axis("off")

    save_path = os.path.join(
        output_folder,
        f"Final_{case_filename.replace('.b2nd','')}_Class{TARGET_CLASS}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Saved: {save_path}")

print("\n ALL DONE")