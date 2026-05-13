import os
import gc

# =====================================================
# 0. NNUNET SETUP
# =====================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["nnUNet_raw"] = "/content/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/content/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/content/nnUNet_results"

import torch
import numpy as np
import matplotlib.pyplot as plt
import blosc2
import torch.nn.functional as F
from tqdm import tqdm
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# =====================================================
# 1. USER SETTINGS
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HARDCODED_RUNS = {

    ("case_35", 3): 107  
}

PREPROCESSED_PATH = "PREPROCESSED_DATA_PATH"
MODEL_FOLDER = "MODEL_PATH"
OUTPUT_ROOT = "OUTPUT_PATH"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

MC_ROUNDS = 20
OCC_CUBE = 32
OCC_STRIDE = 32
SLAB_SIZE = 8
ZOOM_FACTOR = 0.4

print(f" Running 3D XAI Pipeline on {DEVICE}")

# =====================================================
# 2. HELPER: PADDING
# =====================================================
def pad_3d_volume(tensor, multiple=16):
    _, C, D, H, W = tensor.shape
    pad_d = (multiple - D % multiple) % multiple
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    padded = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=tensor.min())
    return padded, (D, H, W)

# =====================================================
# 3. LOAD MODEL
# =====================================================
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=False, use_mirroring=False, device=DEVICE)
predictor.initialize_from_trained_model_folder(
    model_training_output_dir=MODEL_FOLDER,
    use_folds=(0,),
    checkpoint_name="checkpoint_best.pth"
)
model = predictor.network.to(DEVICE).half()
model.eval()

# =====================================================
# 4. PROCESS EACH HARDCODED CASE
# =====================================================
for (case_id, TARGET_CLASS), mid_z_global in HARDCODED_RUNS.items():
    print(f"\n Processing {case_id} | Class: {TARGET_CLASS} | Z-Slice: {mid_z_global}")

    case_output = os.path.join(OUTPUT_ROOT, case_id)
    os.makedirs(case_output, exist_ok=True)

    b2nd_file = os.path.join(PREPROCESSED_PATH, case_id + ".b2nd")
    if not os.path.exists(b2nd_file):
        print(f" Missing {b2nd_file}")
        continue

    # Load Slab
    vol_np = blosc2.open(b2nd_file)[:].astype(np.float32)
    orig_D_total = vol_np.shape[1]

    z_start = max(0, mid_z_global - SLAB_SIZE // 2)
    z_end = min(orig_D_total, z_start + SLAB_SIZE)
    vol_slab = vol_np[:, z_start:z_end, :, :]

    img_slice_np = vol_slab[0, mid_z_global - z_start].copy()

    vol_torch = torch.from_numpy(vol_slab).unsqueeze(0).to(DEVICE).half()

    x_padded, (orig_D, orig_H, orig_W) = pad_3d_volume(vol_torch, multiple=8)
    mid_z_local = mid_z_global - z_start

    del vol_np, vol_slab, vol_torch
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # 3D GRAD-CAM
    # -----------------------------------------------------
    print(" Running Grad-CAM...")
    model.zero_grad()
    activations, gradients = None, None
    def fwd_hook(m, i, o): global activations; activations = o
    def bwd_hook(m, gi, go): global gradients; gradients = go[0]

    target_layer = model.decoder.stages[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)


    x_padded.requires_grad_(False) # STOP tracking the input image!

    
    for name, param in model.named_parameters():
        if "decoder" in name or "seg" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    with torch.amp.autocast('cuda'):
        out = model(x_padded)
        if isinstance(out, (list, tuple)): out = out[0]
        score = out[:, TARGET_CLASS, mid_z_local].mean()

    score.backward()
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    cam3d = torch.relu((weights * activations).sum(dim=1)).squeeze().detach().cpu().numpy()

    h1.remove(); h2.remove(); del out, score

    for param in model.parameters():
        param.requires_grad = False

    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # 3D MC DROPOUT
    # -----------------------------------------------------
    print(f" Running MC Dropout...")
    def enable_dropout_3d(m):
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)): m.train()

    model.apply(enable_dropout_3d)
    preds = []
    for _ in range(MC_ROUNDS):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out_mc = model(x_padded)
                if isinstance(out_mc, (list, tuple)): out_mc = out_mc[0]
                preds.append(torch.softmax(out_mc, dim=1)[:, TARGET_CLASS, mid_z_local].cpu().numpy())
                del out_mc

    uncertainty2d = np.var(np.stack(preds), axis=0)[0]
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # 3D OCCLUSION
    # -----------------------------------------------------
    print("🕵️ Running Occlusion...")
    occ_map = np.zeros((x_padded.shape[3], x_padded.shape[4]))

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            base_out = model(x_padded)
            if isinstance(base_out, (list, tuple)): base_out = base_out[0]
            base_score = torch.softmax(base_out, dim=1)[0, TARGET_CLASS, mid_z_local].mean().item()
            del base_out

    ys = range(0, x_padded.shape[3] - OCC_CUBE + 1, OCC_STRIDE)
    xs = range(0, x_padded.shape[4] - OCC_CUBE + 1, OCC_STRIDE)

    with torch.no_grad():
        for y in tqdm(ys):
            for xc in xs:
                x_occ = x_padded.clone().detach()
                x_occ[:, :, mid_z_local, y:y+OCC_CUBE, xc:xc+OCC_CUBE] = x_padded.min()

                with torch.amp.autocast('cuda'):
                    out_occ = model(x_occ)
                    if isinstance(out_occ, (list, tuple)): out_occ = out_occ[0]
                    new_score = torch.softmax(out_occ, dim=1)[0, TARGET_CLASS, mid_z_local].mean().item()

                occ_map[y:y+OCC_CUBE, xc:xc+OCC_CUBE] += (base_score - new_score)
                del x_occ, out_occ

    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------
    print("Generating PNG...")

    cam_slice = cam3d[mid_z_local]

    h, w = orig_H, orig_W
    y_center, x_center = h // 2, w // 2
    y_size, x_size = int(h * ZOOM_FACTOR), int(w * ZOOM_FACTOR)
    y1, y2 = y_center - y_size // 2, y_center + y_size // 2
    x1, x2 = x_center - x_size // 2, x_center + x_size // 2

    plt.figure(figsize=(20, 5))
    titles = ["A. Original CT", "B. Grad-CAM (Attention)", "C. Uncertainty (Confidence)", "D. Occlusion (Importance)"]
    maps = [None, cam_slice, uncertainty2d, occ_map]

    for i in range(4):
        ax = plt.subplot(1, 4, i+1)
        plt.imshow(img_slice_np, cmap="gray")
        if maps[i] is not None:
            m = maps[i][:orig_H, :orig_W]

            if i == 2:  # ONLY uncertainty
                plt.imshow(m, cmap="jet", alpha=0.5, vmin=0, vmax=0.05)
            else:
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                plt.imshow(m, cmap="jet", alpha=0.5)

        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)

        plt.title(titles[i], fontsize=14, fontweight='bold')
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(case_output, f"Paper_Zoom_{case_id}_Class_{TARGET_CLASS}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    del x_padded, cam_slice, uncertainty2d, occ_map, img_slice_np
    gc.collect()
    torch.cuda.empty_cache()

print("\n Done!")