"""
============================================================
  Medical Physics QA Suite
  Combined tool — all modules in one file
  Modules:
    1.  ACR Phantom QA
    2.  CatPhan CT QA (pylinac)
    3.  Cheese Phantom Gamma QA
    4.  Dose Calibration (TG-51 / TRS-398)
    5.  EPID / Picket Fence QA
    6.  Field Analysis (2D image)
    7.  Field Profile Analysis (1D CSV)
    8.  Trajectory Log Analysis
    9.  MLC Static Field Analysis
    10. Nuclear Medicine QA
    11. Planar Imaging QA (Leeds TOR)
    12. QUART Phantom QA
    13. Starshot QA
    14. VMAT Log QA
    15. Winston-Lutz QA (pylinac)
    16. Winston-Lutz Multi-Target QA (pylinac)
    17. Winston-Lutz Custom Isocenter (OpenCV)
============================================================
"""

# ── Standard library ──────────────────────────────────────
import os
import glob
import math
from datetime import datetime

# ── Third-party ───────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom

# ── Optional / heavy deps (imported inside functions) ─────
# cv2, SimpleITK, scipy, skimage, pylinac, tkinter
# These are imported lazily so missing packages only break
# the specific module that needs them.


# ╔══════════════════════════════════════════════════════════╗
# ║  1. ACR PHANTOM QA                                       ║
# ╚══════════════════════════════════════════════════════════╝

def acr_load_dicom_slice(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    # Use getattr fallbacks — some DICOM files omit these tags
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return pixel_array * slope + intercept


def acr_roi_stats(image, center, radius):
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    roi_pixels = image[mask]
    return np.mean(roi_pixels), np.std(roi_pixels)


def analyze_acr_slice(image_path):
    print(f"📷 Analyzing ACR phantom slice: {image_path}")
    image = acr_load_dicom_slice(image_path)

    plt.imshow(image, cmap='gray')
    plt.title("ACR Phantom Slice")
    plt.colorbar(label="HU")
    plt.show()

    # Example ROIs — verify visually or adapt to your site
    rois = {
        "Water":   (256, 256),
        "Air":     (100, 100),
        "Acrylic": (300, 100),
        "Bone":    (400, 250),
    }
    radius = 10

    print("\n=== ACR Phantom QA Results ===")
    for name, center in rois.items():
        mean, std = acr_roi_stats(image, center, radius)
        print(f"  {name}: Mean HU = {mean:.1f}, SD = {std:.1f}")
    print("==============================")


def run_acr_phantom():
    print("\n=== ACR Phantom QA Tool ===")
    path = input("Enter path to ACR DICOM slice: ").strip()
    if not os.path.isfile(path):
        print("❌ File not found.")
        return
    try:
        analyze_acr_slice(path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  2. CATPHAN CT QA                                        ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_catphan(folder_path):
    from pylinac import CatPhan504  # swap to CatPhan600 if needed
    print(f"📂 Analyzing CatPhan DICOM series from: {folder_path}")

    phantom = CatPhan504(folder_path)
    phantom.analyze()

    print("\n=== CatPhan QA Results ===")
    print(f"  Geometric accuracy: {phantom.ctp404.distances_mm}")
    print(f"  Slice thickness:    {phantom.ctp404.slice_thickness_mm:.2f} mm")
    print(f"  Low-contrast ROIs:  {phantom.ctp515.num_rois}")
    print("  HU values:")
    for material, value in phantom.ctp404.hu_values.items():
        print(f"    - {material}: {value:.1f} HU")
    print("===========================")

    phantom.plot_analyzed_image()
    plt.show()
    phantom.save_analyzed_image("catphan_analysis.png")
    phantom.publish_pdf("catphan_report.pdf")
    print("✅ PDF report and analysis image saved.")


def run_catphan():
    print("\n=== CatPhan CT QA Tool ===")
    path = input("Enter path to CatPhan DICOM folder: ").strip()
    if not os.path.isdir(path):
        print("❌ Invalid folder.")
        return
    try:
        analyze_catphan(path)
    except Exception as e:
        print(f"❌ Analysis error: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  3. CHEESE PHANTOM GAMMA QA                              ║
# ╚══════════════════════════════════════════════════════════╝

def cheese_load_dose(dcm_path):
    import SimpleITK as sitk
    img = sitk.ReadImage(dcm_path)
    arr = sitk.GetArrayFromImage(img)[0]
    spacing = img.GetSpacing()
    return arr, spacing[:2]


def cheese_gamma_index(reference, evaluated,
                       dose_threshold=0.03, dist_threshold=3,
                       pixel_spacing=(1, 1)):
    """Simple global 2-D gamma index."""
    ref_norm  = reference.astype(np.float64) / np.max(reference)
    eval_norm = evaluated.astype(np.float64) / np.max(evaluated)
    # Must use explicit float64 — full_like on an int array silently wraps inf to
    # -2147483648 (confirmed NumPy 2.x behaviour), corrupting the entire gamma map.
    gamma_map = np.full(reference.shape, np.inf, dtype=np.float64)

    for i in range(reference.shape[0]):
        for j in range(reference.shape[1]):
            ref_dose = ref_norm[i, j]
            for di in range(-5, 6):
                for dj in range(-5, 6):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < reference.shape[0] and 0 <= jj < reference.shape[1]:
                        dist = np.sqrt(
                            (di * pixel_spacing[0]) ** 2 +
                            (dj * pixel_spacing[1]) ** 2
                        )
                        dose_diff = abs(ref_dose - eval_norm[ii, jj])
                        g = np.sqrt(
                            (dose_diff / dose_threshold) ** 2 +
                            (dist / dist_threshold) ** 2
                        )
                        gamma_map[i, j] = min(gamma_map[i, j], g)
    return gamma_map


def analyze_cheese_phantom(plan_path, meas_path):
    print("📥 Loading dose distributions...")
    ref_dose, spacing = cheese_load_dose(plan_path)
    meas_dose, _      = cheese_load_dose(meas_path)

    print("⏳ Computing gamma index (this may take a while)...")
    gamma = cheese_gamma_index(ref_dose, meas_dose,
                               dose_threshold=0.03,
                               dist_threshold=3,
                               pixel_spacing=spacing)

    pass_rate = 100 * np.sum(gamma <= 1) / gamma.size
    print("\n=== Cheese Phantom QA Result ===")
    print(f"  Gamma pass rate (3%/3mm): {pass_rate:.2f}%")
    print("================================")

    plt.imshow(gamma, cmap='hot', vmin=0, vmax=2)
    plt.colorbar(label='Gamma Index')
    plt.title("Gamma Map")
    plt.savefig("cheese_gamma_map.png")
    plt.show()
    print("✅ Gamma map saved.")


def run_cheese_phantom():
    print("\n=== Cheese Phantom Gamma QA Tool ===")
    ref_path  = input("Enter planned RTDOSE DICOM path: ").strip()
    eval_path = input("Enter measured RTDOSE DICOM path: ").strip()
    try:
        analyze_cheese_phantom(ref_path, eval_path)
    except Exception as e:
        print(f"❌ Error: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  4. DOSE CALIBRATION (TG-51 / TRS-398)                   ║
# ╚══════════════════════════════════════════════════════════╝

def compute_dose(M, N_Dw, k_Q, P_TP=1.0, P_ion=1.0, P_pol=1.0):
    return M * N_Dw * k_Q * P_TP * P_ion * P_pol


def _get_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("⚠️  Please enter a valid number.")


def run_dose_calibration():
    print("\n=== TG-51 / TRS-398 Dose Calibration Calculator ===")
    M    = _get_float("Electrometer reading M (nC): ")
    N_Dw = _get_float("Chamber calibration factor N_D,w (cGy/nC): ")
    k_Q  = _get_float("Beam quality correction factor k_Q: ")
    P_TP = _get_float("P_TP  (Temperature & Pressure correction) [1.0]: ", 1.0)
    P_ion = _get_float("P_ion (Ion recombination correction)     [1.0]: ", 1.0)
    P_pol = _get_float("P_pol (Polarity correction)              [1.0]: ", 1.0)

    dose = compute_dose(M, N_Dw, k_Q, P_TP, P_ion, P_pol)
    print("\n--- Calibration Result ---")
    print(f"  Dose to water (D_w): {dose:.2f} cGy")
    print("---------------------------")


# ╔══════════════════════════════════════════════════════════╗
# ║  5. EPID / PICKET FENCE QA                               ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_picket_fence(image_path, tolerance=0.5, action_tolerance=0.8, dpi=72):
    from pylinac import PicketFence
    pf = PicketFence(image_path)
    pf.analyze(tolerance=tolerance, action_tolerance=action_tolerance, dpi=dpi)

    print("\n=== Picket Fence QA Results ===")
    print(f"  Pickets detected: {pf.num_pickets}")
    print(f"  MLC errors:       {len(pf.errors)}")
    print(f"  Max error:        {pf.max_error:.2f} mm")
    print(f"  Passed:           {'Yes' if pf.passed else 'No'}")
    print("===============================")

    pf.plot_analyzed_image()
    plt.show()
    pf.publish_pdf("picket_fence_report.pdf")
    print("📄 Report saved: picket_fence_report.pdf")


def run_picket_fence():
    print("\n=== EPID / Picket Fence QA Tool ===")
    path = input("Enter path to EPID image (DICOM/TIFF): ").strip()
    try:
        analyze_picket_fence(path)
    except Exception as e:
        print(f"❌ Error: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  6. FIELD ANALYSIS (2-D image)                           ║
# ╚══════════════════════════════════════════════════════════╝

def _field_flatness(profile):
    mid   = len(profile) // 2
    width = len(profile) // 5
    c     = profile[mid - width: mid + width]
    return (np.max(c) - np.min(c)) / (np.max(c) + np.min(c)) * 100


def _field_symmetry(profile):
    mid   = len(profile) // 2
    left  = profile[:mid]
    right = profile[-mid:][::-1]
    return np.max(np.abs(left - right)) * 100


def analyze_field_image(image_path):
    from skimage.io import imread
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops, label

    print(f"📷 Loading image: {image_path}")
    img = imread(image_path, as_gray=True)
    img = img / np.max(img)

    thresh  = threshold_otsu(img)
    binary  = img > thresh
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=img)

    if not regions:
        print("❌ No field region found!")
        return

    field      = max(regions, key=lambda r: r.area)
    field_sz_x = field.bbox[3] - field.bbox[1]
    field_sz_y = field.bbox[2] - field.bbox[0]
    centroid   = field.centroid

    mid_y      = img.shape[0] // 2
    mid_x      = img.shape[1] // 2
    profile_x  = img[mid_y, :]
    profile_y  = img[:, mid_x]

    print("\n=== Field Analysis Results ===")
    print(f"  Field size (px):     X={field_sz_x}, Y={field_sz_y}")
    print(f"  Centroid (row, col): {centroid}")
    print(f"  Flatness X:          {_field_flatness(profile_x):.2f} %")
    print(f"  Flatness Y:          {_field_flatness(profile_y):.2f} %")
    print(f"  Symmetry X:          {_field_symmetry(profile_x):.2f} %")
    print(f"  Symmetry Y:          {_field_symmetry(profile_y):.2f} %")
    print("================================")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].plot(centroid[1], centroid[0], 'r+', markersize=15)
    ax[0].set_title("Field Image with Centroid")
    ax[1].plot(profile_x, label='X Profile')
    ax[1].plot(profile_y, label='Y Profile')
    ax[1].legend()
    ax[1].set_title("Field Profiles")
    plt.tight_layout()
    plt.savefig("field_analysis_plot.png")
    plt.show()
    print("✅ Field analysis plot saved.")


def run_field_analysis():
    print("\n=== Field Analysis Tool ===")
    path = input("Enter path to 2D field image (TIFF/PNG): ").strip()
    if not os.path.isfile(path):
        print("❌ Invalid image path.")
        return
    try:
        analyze_field_image(path)
    except Exception as e:
        print(f"❌ Analysis error: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  7. FIELD PROFILE ANALYSIS (1-D CSV)                     ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_field_profile(file_path):
    raw = pd.read_csv(file_path, header=None)
    # squeeze() is ambiguous on multi-column CSVs — always take the first column
    if raw.shape[1] > 1:
        print(f"  ⚠️  CSV has {raw.shape[1]} columns; using column 0 as the profile.")
    data    = raw.iloc[:, 0]
    profile = data.to_numpy(dtype=np.float64)
    profile = profile / np.max(profile)

    central = profile[len(profile) // 4: 3 * len(profile) // 4]
    flatness = (
        (np.max(central) - np.min(central)) /
        (np.max(central) + np.min(central)) * 100
    )

    mid      = len(profile) // 2
    left     = profile[:mid]
    right    = profile[-mid:][::-1]
    symmetry = np.max(np.abs(left - right)) * 100

    above_half = np.where(profile >= 0.5)[0]
    if above_half.size == 0:
        fwhm_str = "N/A (profile never reaches 50% of max)"
        fwhm_start = fwhm_end = None
    else:
        fwhm_str = str(above_half[-1] - above_half[0])
        fwhm_start, fwhm_end = above_half[0], above_half[-1]

    print("\n=== Field Profile Analysis ===")
    print(f"  Flatness (central 50%): {flatness:.2f} %")
    print(f"  Symmetry (max L/R):     {symmetry:.2f} %")
    print(f"  FWHM (pixels):          {fwhm_str}")
    print("================================")

    plt.plot(profile, label='Profile')
    plt.axhline(0.5, color='r', linestyle='--', label='Half Max')
    if fwhm_start is not None:
        plt.axvline(fwhm_start, color='g', linestyle=':', label='FWHM start')
        plt.axvline(fwhm_end,   color='g', linestyle=':', label='FWHM end')
    plt.title("Field Profile")
    plt.xlabel("Position (pixels or mm)")
    plt.ylabel("Normalized Dose")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("field_profile_analysis.png")
    plt.show()
    print("✅ Profile plot saved.")


def run_field_profile():
    print("\n=== Field Profile Analyzer ===")
    path = input("Enter path to 1D profile CSV file: ").strip()
    try:
        analyze_field_profile(path)
    except Exception as e:
        print(f"❌ Error: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  8. TRAJECTORY LOG ANALYSIS                              ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_log_file(log_file_path):
    from pylinac import TrajectoryLog
    print(f"📄 Loading log file: {log_file_path}")
    log = TrajectoryLog(log_file_path)
    log.analyze()

    print("\n=== Trajectory Log Analysis ===")
    print(f"  Machine:          {log.header.machine}")
    print(f"  Treatment type:   {log.treatment_type}")
    print(f"  Gantry range:     {log.axis_data.gantry.actual.min():.2f}° "
          f"to {log.axis_data.gantry.actual.max():.2f}°")
    print(f"  Max MLC RMS error:{log.fluence.actual_rms:.3f} mm")
    print(f"  Max MLC deviation:{log.axis_data.mlcs.actual.max():.2f} mm")
    print(f"  Pass status:      {'Pass' if log.passed else 'Fail'}")
    print("================================")

    log.plot_summary()
    plt.show()
    log.save_summary("log_summary.png")
    log.publish_pdf("log_report.pdf")
    print("✅ Summary image and report saved.")


def run_log_analyze():
    print("\n=== Trajectory Log QA Tool ===")
    path = input("Enter path to log file (.bin or .elog): ").strip()
    if not os.path.isfile(path):
        print("❌ Invalid file path.")
        return
    try:
        analyze_log_file(path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  9. MLC STATIC FIELD ANALYSIS                            ║
# ╚══════════════════════════════════════════════════════════╝

def mlc_load_dicom(path):
    ds  = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if 'RescaleSlope'     in ds: img = img * float(ds.RescaleSlope)
    if 'RescaleIntercept' in ds: img = img + float(ds.RescaleIntercept)
    img /= np.max(img)
    return img, ds


def mlc_analyze_static_field(img, threshold=0.5):
    img        = img / np.max(img)
    center_row = img[img.shape[0] // 2, :]
    positions  = np.arange(len(center_row))

    above   = center_row >= threshold
    indices = np.where(above)[0]
    if indices.size == 0:
        raise ValueError(
            f"No pixels meet threshold={threshold} in centre row. "
            "Check image normalisation."
        )
    left_edge  = indices[0]
    right_edge = indices[-1]
    print(f"  Field edges (px): {left_edge} — {right_edge}")
    print(f"  Field size (px):  {right_edge - left_edge}")

    # .astype(int) is safer than dtype=int kwarg across NumPy versions
    rows_to_check = np.linspace(
        img.shape[0] // 4, 3 * img.shape[0] // 4, 5
    ).astype(int)
    leaf_positions = []
    for r in rows_to_check:
        row_profile = img[r, :]
        idx = np.where(row_profile >= threshold)[0]
        if idx.size == 0:
            print(f"  ⚠️  Row {r}: no pixels meet threshold — skipped.")
            continue
        leaf_positions.append((idx[0], idx[-1]))

    return positions, center_row, leaf_positions


def run_mlc_static():
    print("\n=== MLC Static Field Analysis ===")
    path = input("Enter path to EPID static field DICOM: ").strip()
    if not os.path.isfile(path):
        print("❌ File not found.")
        return
    try:
        img, _ = mlc_load_dicom(path)
        positions, profile, leaf_positions = mlc_analyze_static_field(img)

        plt.figure(figsize=(10, 4))
        plt.plot(positions, profile, label="Center Profile")
        plt.axhline(0.5, color='r', linestyle='--', label="50% level")
        plt.title("MLC Static Field Profile")
        plt.xlabel("Pixel Position")
        plt.ylabel("Normalized Dose")
        plt.legend()
        plt.grid()
        plt.show()

        print("\n=== Leaf positions at sampled rows ===")
        for i, (left, right) in enumerate(leaf_positions):
            print(f"  Row {i+1}: Left={left}, Right={right}, Width={right-left} px")
    except Exception as e:
        print(f"❌ Analysis error: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  10. NUCLEAR MEDICINE QA                                 ║
# ╚══════════════════════════════════════════════════════════╝

HALF_LIVES = {
    "Tc-99m": 6.01,
    "F-18":   1.83,
    "I-131":  192.5,
    "Ga-68":  1.13,
    "In-111": 67.2,
}


def decay_corrected_activity(A0, elapsed_hr, half_life_hr):
    return A0 * math.exp(-0.693 * elapsed_hr / half_life_hr)


def counting_uncertainty(counts):
    sd = math.sqrt(counts)
    return sd, sd / counts * 100


def dose_constancy_check(measured, reference, tolerance=10.0):
    dev = abs(measured - reference) / reference * 100
    return dev, dev <= tolerance


def run_nuclear():
    print("\n=== Nuclear Medicine QA Tool ===")

    print("\n--- Dose Calibrator Constancy ---")
    ref  = _get_float("Reference activity (MBq): ")
    meas = _get_float("Measured activity  (MBq): ")
    dev, passed = dose_constancy_check(meas, ref)
    print(f"  Deviation: {dev:.2f}%  →  {'PASS ✅' if passed else 'FAIL ❌'}")

    print("\n--- Counting Statistics ---")
    counts = int(_get_float("Total counts (well counter / gamma camera): "))
    sd, pct = counting_uncertainty(counts)
    print(f"  Std deviation: {sd:.2f} counts")
    print(f"  Uncertainty:   {pct:.2f}%")

    print("\n--- Decay Correction ---")
    isotope = input(f"Isotope {list(HALF_LIVES.keys())}: ").strip()
    if isotope not in HALF_LIVES:
        print("⚠️  Isotope not recognized.")
        return
    A0     = _get_float("Original activity (MBq): ")
    t0_str = input("Calibration time (YYYY-MM-DD HH:MM): ")
    t1_str = input("Current time      (YYYY-MM-DD HH:MM): ")
    t0     = datetime.strptime(t0_str, "%Y-%m-%d %H:%M")
    t1     = datetime.strptime(t1_str, "%Y-%m-%d %H:%M")
    dt_hr  = (t1 - t0).total_seconds() / 3600
    A_corr = decay_corrected_activity(A0, dt_hr, HALF_LIVES[isotope])
    print(f"  Decay-corrected activity: {A_corr:.2f} MBq")
    print("\n✅ Nuclear QA complete.")


# ╔══════════════════════════════════════════════════════════╗
# ║  11. PLANAR IMAGING QA (Leeds TOR)                       ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_planar_image(image_path):
    from pylinac import LeedsTOR
    print(f"📷 Analyzing planar image: {image_path}")
    leeds = LeedsTOR(image_path)
    leeds.analyze()

    print("\n=== Planar Imaging QA (Leeds TOR) ===")
    print(f"  Phantom center: ({leeds.center.x:.1f}, {leeds.center.y:.1f})")
    print(f"  Max contrast:   {leeds.max_contrast:.2f}")
    print(f"  Visible disks:  {leeds.num_visible_dots}")
    print(f"  MTF50:          {leeds.mtf.mtf50:.3f}")
    print(f"  Pass status:    {'PASS ✅' if leeds.passed else 'FAIL ❌'}")
    print("======================================")

    leeds.plot_analyzed_image()
    plt.show()
    leeds.save_analyzed_image("planar_image_analysis.png")
    leeds.publish_pdf("planar_image_report.pdf")
    print("✅ Report and image saved.")


def run_planar_imaging():
    print("\n=== Planar Imaging QA Tool (Leeds TOR) ===")
    path = input("Enter path to planar image (DICOM/TIFF): ").strip()
    if not os.path.isfile(path):
        print("❌ Invalid file path.")
        return
    try:
        analyze_planar_image(path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  12. QUART PHANTOM QA                                    ║
# ╚══════════════════════════════════════════════════════════╝

def quart_load_dicom(path):
    ds    = pydicom.dcmread(path)
    img   = ds.pixel_array.astype(np.float32)
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    img = img * slope + intercept
    return img


def quart_roi_stats(img, center, radius=10):
    y, x  = np.ogrid[:img.shape[0], :img.shape[1]]
    mask  = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    pixels = img[mask]
    return np.mean(pixels), np.std(pixels)


def analyze_quart_image(dcm_path):
    print(f"📷 Analyzing QUART phantom image: {dcm_path}")
    img = quart_load_dicom(dcm_path)

    # Adjust coordinates for your CBCT setup
    rois = {
        "Center": (256, 256),
        "Top":    (256, 100),
        "Bottom": (256, 412),
        "Left":   (100, 256),
        "Right":  (412, 256),
    }

    print("\n=== QUART Phantom QA ===")
    center_mean, _ = quart_roi_stats(img, rois["Center"])
    uniformity_devs = []

    for name, center in rois.items():
        mean, std = quart_roi_stats(img, center)
        print(f"  {name}: Mean HU = {mean:.1f}, StdDev = {std:.1f}")
        if name != "Center":
            uniformity_devs.append(abs(mean - center_mean))

    print(f"\n  Uniformity deviation from center: {max(uniformity_devs):.1f} HU")
    print("===============================")

    plt.imshow(img, cmap='gray')
    for name, (x, y) in rois.items():
        plt.plot(x, y, 'ro')
        plt.text(x + 5, y, name, color='red')
    plt.title("QUART Phantom with ROIs")
    plt.colorbar(label='HU')
    plt.savefig("quart_analysis.png")
    plt.show()
    print("✅ Image and ROI map saved.")


def run_quart():
    print("\n=== QUART Phantom QA Tool ===")
    path = input("Enter path to QUART DICOM image: ").strip()
    if not os.path.isfile(path):
        print("❌ Invalid path.")
        return
    try:
        analyze_quart_image(path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  13. STARSHOT QA                                         ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_starshot(image_path):
    from pylinac import Starshot
    print(f"📷 Loading starshot image: {image_path}")
    star = Starshot(image_path)
    star.analyze(tolerance=1.0, dpi=100)

    print("\n=== Starshot QA Results ===")
    print(f"  Isocenter: x={star.center.x:.2f} mm, y={star.center.y:.2f} mm")
    print(f"  Circle radius: {star.radius_mm:.2f} mm")
    print(f"  Pass status: {'PASS ✅' if star.passed else 'FAIL ❌'}")
    print("============================")

    star.plot_analyzed_image()
    plt.show()
    star.save_analyzed_image("starshot_result.png")
    star.publish_pdf("starshot_report.pdf")
    print("✅ Report and analyzed image saved.")


def run_starshot():
    print("\n=== Starshot QA Tool ===")
    path = input("Enter path to starshot image (DICOM/TIFF): ").strip()
    if not os.path.isfile(path):
        print("❌ Invalid image path.")
        return
    try:
        analyze_starshot(path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  14. VMAT LOG QA                                         ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_vmat_log(log_file_path):
    from pylinac import TrajectoryLog
    print(f"📄 Analyzing VMAT log file: {log_file_path}")
    log = TrajectoryLog(log_file_path)
    log.analyze()

    print("\n=== VMAT QA Results (Log File) ===")
    print(f"  Machine:       {log.header.machine}")
    print(f"  Delivery type: {log.treatment_type}")
    print(f"  Gantry range:  {log.axis_data.gantry.actual.min():.2f}° "
          f"to {log.axis_data.gantry.actual.max():.2f}°")
    print(f"  MLC RMS error: {log.fluence.actual_rms:.3f} mm")
    print(f"  Max MLC error: {log.axis_data.mlcs.actual.max():.3f} mm")
    print(f"  Pass status:   {'PASS ✅' if log.passed else 'FAIL ❌'}")
    print("===================================")

    log.plot_summary()
    plt.show()
    log.save_summary("vmat_log_summary.png")
    log.publish_pdf("vmat_log_report.pdf")
    print("✅ Summary image and PDF report saved.")


def run_vmat():
    print("\n=== VMAT QA Tool ===")
    path = input("Enter path to VMAT log file (.bin or .elog): ").strip()
    if not os.path.isfile(path):
        print("❌ Invalid file path.")
        return
    try:
        analyze_vmat_log(path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  15. WINSTON-LUTZ QA (pylinac)                           ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_winston_lutz(image_folder):
    from pylinac import WinstonLutz
    print(f"📂 Loading WL images from: {image_folder}")
    wl = WinstonLutz(image_folder)
    wl.analyze()

    print("\n=== Winston-Lutz QA Results ===")
    print(f"  Max 2D CAX-to-BB distance: {wl.max_2D_distance:.2f} mm")
    print(f"  Max 3D vector deviation:   {wl.max_3D_distance:.2f} mm")
    print(f"  Isocenter diameter (3D):   {wl.isocenter_diameter:.2f} mm")
    print("===============================")

    wl.plot_summary()
    plt.show()
    wl.save_summary("winston_lutz_summary.png")
    wl.publish_pdf("winston_lutz_report.pdf")
    print("✅ Report and image saved.")


def run_winston_lutz():
    print("\n=== Winston-Lutz QA Tool ===")
    path = input("Enter path to folder with WL DICOM images: ").strip()
    if not os.path.isdir(path):
        print("❌ Invalid folder path.")
        return
    try:
        analyze_winston_lutz(path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  16. WINSTON-LUTZ MULTI-TARGET QA (pylinac)              ║
# ╚══════════════════════════════════════════════════════════╝

def analyze_multi_target_wl(image_folder):
    from pylinac import WinstonLutz
    print(f"📂 Loading images from: {image_folder}")
    wl = WinstonLutz(image_folder)
    wl.analyze()

    print("\n=== Multi-Target Winston-Lutz QA Results ===")
    for beam in wl.beams:
        print(f"\n  📸 {os.path.basename(beam.image.path)}")
        print(f"     Gantry={beam.gantry_angle}, "
              f"Coll={beam.collimator_angle}, "
              f"Couch={beam.table_angle}")
        print(f"     BBs detected: {len(beam.bb_objs)}")
        for i, bb in enumerate(beam.bb_objs, 1):
            # cax2bb_vector is a 3-element vector, not a scalar — use its magnitude
            vec = bb.cax2bb_vector
            vec_mag = float(np.linalg.norm(vec)) if hasattr(vec, '__len__') else float(vec)
            print(f"       BB {i}: 2D={bb.cax2bb_distance:.3f} mm  "
                  f"3D vector magnitude={vec_mag:.3f} mm")

    print(f"\n  Max 2D deviation: {wl.max_2D_distance:.3f} mm")
    print(f"  Max 3D deviation: {wl.max_3D_distance:.3f} mm")
    print(f"  Isocenter sphere: {wl.isocenter_diameter:.3f} mm")

    wl.plot_summary()
    plt.show()
    wl.save_summary("multi_target_wl_summary.png")
    wl.publish_pdf("multi_target_wl_report.pdf")
    print("✅ Summary image and PDF report saved.")


def run_wl_multi_target():
    print("\n=== Multi-Target Winston-Lutz QA ===")
    use_dialog = input("Use file dialog to select folder? (y/n): ").strip().lower()
    if use_dialog == 'y':
        try:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(
                title="Select WL DICOM Images Folder"
            )
            root.destroy()
        except Exception:
            folder = ""
        if not folder:
            print("❌ No folder selected.")
            return
    else:
        folder = input("Enter path to WL DICOM images folder: ").strip()

    if not os.path.isdir(folder):
        print("❌ Invalid folder path.")
        return
    try:
        analyze_multi_target_wl(folder)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  17. WINSTON-LUTZ CUSTOM ISOCENTER (OpenCV)              ║
# ╚══════════════════════════════════════════════════════════╝

def wl_load_dicom(path):
    ds  = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if 'RescaleSlope'     in ds: img = img * float(ds.RescaleSlope)
    if 'RescaleIntercept' in ds: img = img + float(ds.RescaleIntercept)
    img /= np.max(img)
    return img, ds


def detect_bb(img):
    import cv2
    img_uint8 = (img * 255).astype(np.uint8)
    blurred   = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    circles   = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20,
        param1=50, param2=15,
        minRadius=2, maxRadius=20
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[0][:2]   # x, y of first hit
    raise Exception("❌ BB not detected in image.")


def analyze_custom_isocenter(folder):
    files     = sorted(glob.glob(os.path.join(folder, "*")))
    centers   = []
    filenames = []
    px_spacing = None

    print(f"🔍 Processing {len(files)} files...")
    for path in files:
        try:
            img, ds = wl_load_dicom(path)
            bb_pos  = detect_bb(img)
            centers.append(bb_pos)
            filenames.append(os.path.basename(path))
            if px_spacing is None and hasattr(ds, "PixelSpacing"):
                px_spacing = float(ds.PixelSpacing[0])
            print(f"  ✅ {os.path.basename(path)}: BB at {bb_pos}")
        except Exception as e:
            print(f"  ⚠️  {os.path.basename(path)}: {e}")

    if not centers:
        print("❌ No BB positions found.")
        return

    centers     = np.array(centers)
    mean_center = np.mean(centers, axis=0)
    deviations  = np.linalg.norm(centers - mean_center, axis=1)
    max_dev     = np.max(deviations)
    sphere_px   = 2 * max_dev

    print("\n=== Isocenter Analysis Results ===")
    print(f"  Mean center (px):   {mean_center}")
    print(f"  Max deviation (px): {max_dev:.2f}")
    print(f"  Sphere diam  (px):  {sphere_px:.2f}")

    if px_spacing:
        print(f"  Max deviation (mm): {max_dev * px_spacing:.2f}")
        print(f"  Sphere diam  (mm):  {sphere_px * px_spacing:.2f}")
        deviations_mm = deviations * px_spacing
    else:
        print("  ⚠️  PixelSpacing not found — mm conversion unavailable.")
        deviations_mm = None

    df = pd.DataFrame({
        "Image":        filenames,
        "X (px)":       centers[:, 0],
        "Y (px)":       centers[:, 1],
        "Deviation (px)": deviations,
    })
    if deviations_mm is not None:
        df["Deviation (mm)"] = deviations_mm

    df.to_csv("isocenter_analysis_results.csv", index=False)
    print("📄 Results saved to isocenter_analysis_results.csv")

    plt.figure(figsize=(6, 6))
    plt.scatter(centers[:, 0], centers[:, 1], label="BB Centers")
    plt.scatter(mean_center[0], mean_center[1], c='r', label="Mean Center")
    for i, fname in enumerate(filenames):
        plt.annotate(str(i + 1), (centers[i, 0], centers[i, 1]), fontsize=8)
    plt.title("Isocenter Analysis (Winston–Lutz, custom)")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("isocenter_analysis_plot.png")
    plt.show()
    print("🖼  Plot saved to isocenter_analysis_plot.png")
    print("\n=== Detailed Results ===")
    print(df.to_string(index=False))


def run_wl_custom():
    print("\n=== Winston-Lutz Custom Isocenter (OpenCV) ===")
    folder = input("Enter folder with WL DICOM images: ").strip()
    if not os.path.isdir(folder):
        print("❌ Invalid folder path.")
        return
    try:
        analyze_custom_isocenter(folder)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN MENU                                               ║
# ╚══════════════════════════════════════════════════════════╝

MENU = [
    ("ACR Phantom QA",                        run_acr_phantom),
    ("CatPhan CT QA (pylinac)",               run_catphan),
    ("Cheese Phantom Gamma QA",               run_cheese_phantom),
    ("Dose Calibration (TG-51 / TRS-398)",    run_dose_calibration),
    ("EPID / Picket Fence QA",                run_picket_fence),
    ("Field Analysis (2D image)",             run_field_analysis),
    ("Field Profile Analysis (1D CSV)",       run_field_profile),
    ("Trajectory Log Analysis",               run_log_analyze),
    ("MLC Static Field Analysis",             run_mlc_static),
    ("Nuclear Medicine QA",                   run_nuclear),
    ("Planar Imaging QA (Leeds TOR)",         run_planar_imaging),
    ("QUART Phantom QA",                      run_quart),
    ("Starshot QA",                           run_starshot),
    ("VMAT Log QA",                           run_vmat),
    ("Winston-Lutz QA (pylinac)",             run_winston_lutz),
    ("Winston-Lutz Multi-Target (pylinac)",   run_wl_multi_target),
    ("Winston-Lutz Custom Isocenter (OpenCV)",run_wl_custom),
]


def main():
    while True:
        print("\n" + "=" * 55)
        print("       Medical Physics QA Suite")
        print("=" * 55)
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i:>2}. {label}")
        print("   0. Exit")
        print("=" * 55)

        choice = input("Select a module [0-17]: ").strip()

        if choice == "0":
            print("👋 Goodbye.")
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(MENU)):
            print("⚠️  Invalid choice. Please enter a number between 0 and 17.")
            continue

        _, run_func = MENU[int(choice) - 1]
        try:
            run_func()
        except KeyboardInterrupt:
            print("\n⏸  Module interrupted. Returning to menu...")


if __name__ == "__main__":
    main()

