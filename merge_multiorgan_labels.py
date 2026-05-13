# Merges individual organ segmentation masks into a single multi-class label map for nnU-Net training.
import os
import glob
import SimpleITK as sitk
import numpy as np

RAW_DIR = "/path/to/HaNSeg_dataset"
LABELS_TR_DIR = "/path/to/labelsTr"
os.makedirs(LABELS_TR_DIR, exist_ok=True)

LABEL_MAP = {
    "SpinalCord": 1,
    "Brainstem": 2,
    "Bone_Mandible": 3,
    "Parotid_L": 4,
    "Parotid_R": 5,
    "A_Carotid_L": 6,
    "A_Carotid_R": 7,
    "Arytenoid": 8,
    "BuccalMucosa": 9,
    "Cavity_Oral": 10,
    "Cochlea_L": 11,
    "Cochlea_R": 12,
    "Cricopharyngeus": 13,
    "Esophagus_S": 14,
    "Eye_AL": 15,
    "Eye_AR": 16,
    "Eye_PL": 17,
    "Eye_PR": 18,
    "Glnd_Lacrimal_L": 19,
    "Glnd_Lacrimal_R": 20,
    "Glnd_Submand_L": 21,
    "Glnd_Submand_R": 22,
    "Glnd_Thyroid": 23,
    "Glottis": 24,
    "Larynx_SG": 25,
    "Lips": 26,
    "OpticChiasm": 27,
    "OpticNrv_L": 28,
    "OpticNrv_R": 29,
    "Pituitary": 30
}

cases = sorted([d for d in os.listdir(RAW_DIR) if d.startswith("case_")])

for case in cases:
    print(f"\nProcessing {case}")

    case_dir = os.path.join(RAW_DIR, case)

    ct_file = glob.glob(os.path.join(case_dir, "*IMG_CT*.nrrd"))
    if not ct_file:
        print(" CT missing, skipping")
        continue

    ct_img = sitk.ReadImage(ct_file[0])
    ct_arr = sitk.GetArrayFromImage(ct_img)

    merged = np.zeros(ct_arr.shape, dtype=np.uint8)

    for organ, label in LABEL_MAP.items():
        seg = glob.glob(os.path.join(case_dir, f"*{organ}*.seg.nrrd"))
        if not seg:
            continue

        seg_img = sitk.ReadImage(seg[0])
        seg_img = sitk.Resample(
            seg_img, ct_img, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )

        seg_arr = sitk.GetArrayFromImage(seg_img)
        merged[seg_arr > 0] = label

    out = sitk.GetImageFromArray(merged)
    out.CopyInformation(ct_img)

    sitk.WriteImage(out, os.path.join(LABELS_TR_DIR, f"{case}.nrrd"))
    print(f"Saved {case}.nrrd")

print("\n DONE: All organs merged")
