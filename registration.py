#pip install SimpleITK numpy matplotlib antspyx itk-elastix
#python -c "import ants; print(ants.__version__)"  
#python -c "import SimpleITK, numpy, matplotlib;"
"""
Automation: run rigid registration (SimpleITK and optionally ANTsPy) per patient folder.
Outputs:
 - OUT_DIR/<patient>/<method>/MR_registered.nii.gz
 - OUT_DIR/<patient>/qa/<method>/axial.png, coronal.png, sagittal.png, report.html
 - OUT_DIR/automation_rigid_log.csv
Config: edit RAW_DIR, OUT_DIR, METHODS, TEST_PATIENT_LIST, ANTS_WHITELIST as needed.
"""
import os, glob, tempfile, shutil, traceback
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


import csv
from datetime import datetime

class PatientLogger:
    def __init__(self, out_dir):
        self.log_file = os.path.join(out_dir, "automation_rigid_log.csv")
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "patient_id",
                    "status",
                    "ct_file",
                    "mr_file",
                    "message"
                ])

    def log(self, patient_id, status, ct_path=None, mr_path=None, message=""):
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                patient_id,
                status,
                os.path.basename(ct_path) if ct_path else "",
                os.path.basename(mr_path) if mr_path else "",
                message
            ])


RAW_DIR = "/path/to/HaNSeg_dataset"
OUT_DIR = "/path/to/nnUNet_imagesTr"
GENERATE_QC = False


TEST_PATIENT_LIST = []  

SKIP_IF_EXISTS = True   

# ---------- helpers ----------
def safe_write_image(img, out_path, use_compression=True):
    out_dir = os.path.dirname(out_path)
    try:
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print("Warning: could not create output directory:", out_dir, e)
    try:
        sitk.WriteImage(img, out_path, useCompression=use_compression)
        return out_path
    except Exception as e:
        print("Direct write failed:", e)
    try:
        base_name = os.path.basename(out_path)
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, base_name)
        sitk.WriteImage(img, tmp_path, useCompression=use_compression)
        try:
            shutil.move(tmp_path, out_path)
            return out_path
        except Exception as mv_err:
            print("Fallback move failed:", mv_err)
            return tmp_path
    except Exception as e2:
        traceback.print_exc()
        raise RuntimeError("Failed to save image.") from e2

# ---------- rigid registration functions ----------
def run_sitk_rigid(fixed, moving):
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.02)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4,
                                                 numberOfIterations=200, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = reg.Execute(fixed, moving)
    registered = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return registered, final_transform, reg


# ---------- per-patient processing ----------
def find_ct_mr(patient_folder):
    candidates = sorted(glob.glob(os.path.join(patient_folder, "*.nrrd")))
    ct = None
    mr = None

    for c in candidates:
        nm = os.path.basename(c).lower()
        if "img_ct" in nm:
            ct = c
        if "img_mr" in nm or "mr_t1" in nm:
            mr = c

    return ct, mr

# ---------- Robust QC helpers (clamped indices) ----------
def _get_center_indices_safe(fixed_img, reg_img):
    fa = sitk.GetArrayFromImage(fixed_img)   # z,y,x
    ra = sitk.GetArrayFromImage(reg_img)
    fz, fy, fx = fa.shape
    # handle possible lower-dim arrays
    rz, ry, rx = ra.shape if ra.ndim==3 else (fz, fy, fx)
    ax_idx = min(fz // 2, rz // 2)
    co_idx = min(fy // 2, ry // 2)
    sa_idx = min(fx // 2, rx // 2)
    return int(ax_idx), int(co_idx), int(sa_idx)

def _get_slice_pair(fixed_img, reg_img, plane, idx):
    fa = sitk.GetArrayFromImage(fixed_img)
    ra = sitk.GetArrayFromImage(reg_img)
    if plane == 'axial':
        return fa[idx, :, :], ra[idx, :, :]
    if plane == 'coronal':
        return fa[:, idx, :], ra[:, idx, :]
    return fa[:, :, idx], ra[:, :, idx]

def _rescale_for_display(slice2d):
    a = slice2d.astype(float)
    mask = np.isfinite(a)
    if not mask.any():
        return np.zeros_like(a, dtype=float)
    lo, hi = np.percentile(a[mask], (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    out = (a - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out

def _save_overlay(fixed_slice, reg_slice, out_png, alpha=0.45):
    bg = _rescale_for_display(fixed_slice)
    fg = _rescale_for_display(reg_slice)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(bg, cmap="gray", aspect='equal')
    ax.imshow(fg, cmap="hot", alpha=alpha, aspect='equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def make_qc_images(patient_id, fixed_img, registered_img, out_root):
    out_dir = Path(out_root) / patient_id 
    out_dir.mkdir(parents=True, exist_ok=True)

    ax_idx, co_idx, sa_idx = _get_center_indices_safe(fixed_img, registered_img)

    try:
        f_ax, r_ax = _get_slice_pair(fixed_img, registered_img, 'axial', ax_idx)
        f_co, r_co = _get_slice_pair(fixed_img, registered_img, 'coronal', co_idx)
        f_sa, r_sa = _get_slice_pair(fixed_img, registered_img, 'sagittal', sa_idx)
    except Exception:
        fa = sitk.GetArrayFromImage(fixed_img)
        f_ax = fa[fa.shape[0]//2,:,:]; f_co = fa[:,fa.shape[1]//2,:]; f_sa = fa[:,:,fa.shape[2]//2]
        ra = sitk.GetArrayFromImage(registered_img)
        r_ax = ra[ra.shape[0]//2,:,:] if ra.ndim==3 else f_ax
        r_co = ra[:,ra.shape[1]//2,:] if ra.ndim==3 else f_co
        r_sa = ra[:,:,ra.shape[2]//2] if ra.ndim==3 else f_sa

    p_ax = out_dir / "axial.png"
    p_co = out_dir / "coronal.png"
    p_sa = out_dir / "sagittal.png"

    _save_overlay(f_ax, r_ax, str(p_ax))
    _save_overlay(f_co, r_co, str(p_co))
    _save_overlay(f_sa, r_sa, str(p_sa))

    html = f"""<html><body>
    <h3>QC Report — {patient_id}</h3>
    <p>Indices used (axial, coronal, sagittal): {ax_idx}, {co_idx}, {sa_idx}</p>
    <p>Axial (center):<br><img src="axial.png" width=600></p>
    <p>Coronal (center):<br><img src="coronal.png" width=600></p>
    <p>Sagittal (center):<br><img src="sagittal.png" width=600></p>
    <p>Inspect CT (background) vs MR_registered (overlay)</p>
    </body></html>"""
    (out_dir / "report.html").write_text(html, encoding='utf-8')

    return str(out_dir)

# ---------- main processing per patient ----------
def process_patient(patient_folder, logger=None):
    patient_id = os.path.basename(patient_folder.rstrip("/\\"))
    ct_path, mr_path = find_ct_mr(patient_folder)

    if not ct_path or not mr_path:
        print(f"[SKIP] {patient_id}")
        print("  CT:", ct_path)
        print("  MR:", mr_path)
        if logger:
            logger.log(patient_id, "SKIPPED", ct_path, mr_path, "Missing CT or MR")
        return

    try:
        fixed = sitk.ReadImage(ct_path, sitk.sitkFloat32)
        moving = sitk.ReadImage(mr_path, sitk.sitkFloat32)

        imagesTr_dir = OUT_DIR
        os.makedirs(imagesTr_dir, exist_ok=True)

        ct_out = os.path.join(
            imagesTr_dir,
            f"{patient_id}_0000.nrrd"
        )
        mr_out = os.path.join(
            imagesTr_dir,
            f"{patient_id}_0001.nrrd"
        )

        # Save CT once
        if not os.path.exists(ct_out):
            safe_write_image(fixed, ct_out)

        # Register MR → CT
        if SKIP_IF_EXISTS and os.path.exists(mr_out):
            registered = sitk.ReadImage(mr_out, sitk.sitkFloat32)
        else:
            registered, _, _ = run_sitk_rigid(fixed, moving)
            safe_write_image(registered, mr_out)

        if logger:
            logger.log(
            patient_id,
            "SUCCESS",
            ct_path,
            mr_path,  
            f"Saved as {os.path.basename(ct_out)}, {os.path.basename(mr_out)}"
            )

    except Exception as e:
        traceback.print_exc()
        if logger:
            logger.log(patient_id, "FAILED", ct_path, mr_path, str(e))

# ---------------- main ----------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    all_patient_folders = sorted([
    d for d in glob.glob(os.path.join(RAW_DIR, "*"))
    if os.path.isdir(d)
])

    if TEST_PATIENT_LIST:
        # build full paths for listed patients, warn if missing
        patient_folders = []
        for pid in TEST_PATIENT_LIST:
            full = os.path.join(RAW_DIR, pid)
            if os.path.isdir(full):
                patient_folders.append(full)
            else:
                print(f"Warning: listed patient folder not found: {full}")
    else:
        patient_folders = all_patient_folders

    if not patient_folders:
        print("No patient folders selected or found in RAW_DIR:", RAW_DIR)
    else:
        print("Patients to process:", [os.path.basename(p) for p in patient_folders])

    logger = PatientLogger(OUT_DIR)

    for pf in patient_folders:
        process_patient(pf, logger=logger)



