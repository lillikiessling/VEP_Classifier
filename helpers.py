
import numpy as np
import matplotlib.pyplot as plt
import os
import re 
import pandas as pd
from tqdm import tqdm


BASE_DIR = "Preprocessed_VEP_Data"
DEVICES = ["PRIMA"]
LABELS = ["BC_Only", "RGC_Only", "BC_and_RGC"]
fs = 2000

def parse_vep_filename(filename):
    # Remove extension and path
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]

    # Pattern now allows underscores or dots in decimals
    pattern = (
        r"(?P<device>[A-Za-z]+\d+)"           # device name (e.g., PRIMA100)
        r"(?:_(?P<trial>\d+))?"                # optional trial index
        r"_+(?P<pulse>[\d_\.]+)ms"            # pulse duration (e.g., 3ms, 0_5ms, 0.5ms)
        r"_+(?P<power>[\d_\.]+)mWmm2"         # irradiance (e.g., 0.3mWmm2, 2.22mWmm2)
    )

    match = re.search(pattern, name)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    parsed = match.groupdict()

    # Clean up underscores in numeric values
    def clean_num(s):
        return float(s.replace("_", ".")) if s else None

    return {
        "device": parsed.get("device"),
        "trial_number": int(parsed["trial"]) if parsed.get("trial") else None,
        "pulse_width_ms": clean_num(parsed.get("pulse")),
        "irradiance_mWmm2": clean_num(parsed.get("power")),
        "filename": base,
    }


def preprocess_signal(time, signal, t_min, t_max, normalize=True):
    """
    Trim signal to [t_min, t_max] ms and optionally z-normalize it.
    """
    mask = (time >= t_min) & (time <= t_max)
    time_trimmed = time[mask]
    signal_trimmed = signal[mask]

    # if normalize: 
    #     # normalize the peak amplitude within the first 50 ms
    #     peak_amplitude = np.max(np.abs(signal_trimmed[time_trimmed <= 50]))
    #     signal_trimmed = signal_trimmed / peak_amplitude if peak_amplitude != 0 else signal_trimmed
    # if normalize:
    #     # min-max normalization to [0, 1]
    #     s_min, s_max = np.min(signal_trimmed), np.max(signal_trimmed)
    #     signal_trimmed = (signal_trimmed - s_min) / (s_max - s_min)
    # z-score normalization
    if normalize:
        signal_trimmed = (signal_trimmed - np.mean(signal_trimmed)) / np.std(signal_trimmed)
        signal_trimmed = np.clip(signal_trimmed, -3, 3)  
    return time_trimmed, signal_trimmed


def load_vep_csv(file_path, t_min=0, t_max=125, normalize=True):
    data = np.loadtxt(file_path, delimiter=",")
    time = data[:, 0]
    signal = data[:, 1]
    time, signal = preprocess_signal(time, signal, t_min=t_min, t_max=t_max, normalize=normalize)
    return time, signal



def load_dataset_paths(base_dir=BASE_DIR, devices=DEVICES, labels=LABELS, ext=".csv"):
    """
    Returns a nested dictionary with all file paths per device and label.
    Example:
        paths["PRIMA"]["BC_Only"] -> list of CSV file paths
    """
    paths = {}
    for device in devices:
        device_dir = os.path.join(base_dir, device)
        paths[device] = {}
        for label in labels:
            dir_path = os.path.join(device_dir, label)
            file_list = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(ext)
            ]
            paths[device][label] = file_list
    return paths

def compute_class_average_signal(file_list, t_min=0, t_max=125, normalize=True):
    """Compute the average signal across *all* files in a given class."""
    all_times, all_signals = [], []

    for file in tqdm(file_list, desc="Computing class average", leave=False):
        time, signal = load_vep_csv(file, t_min=t_min, t_max=t_max, normalize=normalize)
        all_times.append(time)
        all_signals.append(signal)

    # Find overlapping time region across all signals
    common_t_min = max(t[0] for t in all_times)
    common_t_max = min(t[-1] for t in all_times)
    common_time = np.linspace(common_t_min, common_t_max, 1000)

    interpolated_signals = [
        np.interp(common_time, t, s) for t, s in zip(all_times, all_signals)
    ]

    avg_signal = np.mean(interpolated_signals, axis=0)
    return common_time, avg_signal


def summarize_results_and_save(average_results, model_name = "Model"):
    """
    Summarizes average classification results (single or multiple setups),
    focusing only on F1 metrics, sorts by F1_mean descending,
    and saves to CSV with EU-style formatting.

    Automatically handles:
      - average_results as dict of dicts (multiple feature types)
      - average_results as a flat dict (single model)
    """
    output_path= f"results/{model_name}_average_classification_results.csv"
    rows = []
    if isinstance(average_results, dict) and all(isinstance(v, dict) for v in average_results.values()):
        # Case: multiple configurations
        for key, metrics in average_results.items():
            row = {"Model": model_name}

            # --- Extract experimental parameters ---
            if isinstance(key, dict):
                # e.g. average_results[{ "Feature_Type": "Raw", "Dim_Reduction": "PCA", "n_components": 50 }]
                row.update(key)
            elif isinstance(key, tuple):
                # If tuple, infer labels from element types or naming conventions
                # Example: ('Raw', 'PCA', 50) or ('Raw',)
                for item in key:
                    if isinstance(item, str):
                        if any(x in item.lower() for x in ["pca", "ica", "lda", "umap", "kernel"]):
                            row["Dim_Reduction"] = item
                        elif any(x in item.lower() for x in ["raw", "fft", "dwt", "combined", "time", "signal"]):
                            row["Feature_Type"] = item
                        else:
                            # Fallback for custom string
                            row.setdefault("Condition", item)
                    elif isinstance(item, (int, float)):
                        row["n_components"] = item
            elif isinstance(key, str):
                row["Feature_Type"] = key
            else:
                # fallback if weird key type
                row["Condition"] = str(key)

            row.update(metrics)
            rows.append(row)
    else:
        # Case: single setup (flat dict)
        row = {"Model": model_name, **average_results}
        rows.append(row)

    avg_results_df = pd.DataFrame(rows)
    
    # round avlues
    avg_results_df = avg_results_df.round(3)

    # change point to comma so that it is EU style
    def fmt(mean, std):
        return f"{str(mean).replace('.', ',')} ± {str(std).replace('.', ',')}"

    # Build combined DataFrame 
    metrics = {
        "F1 (mean ± std)": ("F1_mean", "F1_std"),
        "BC_Only (mean ± std)": ("Acc_BC_Only_mean", "Acc_BC_Only_std"),
        "RGC_Only (mean ± std)": ("Acc_RGC_Only_mean", "Acc_RGC_Only_std"),
        "BC_and_RGC (mean ± std)": ("Acc_BC_and_RGC_mean", "Acc_BC_and_RGC_std"),
    }
    id_cols = [col for col in ["Model", "Feature_Type", "Dim_Reduction", "n_components", "Condition"] if col in avg_results_df.columns]
    combined_data = {col: avg_results_df[col] for col in id_cols}

    for label, (mean_col, std_col) in metrics.items():
        if mean_col in avg_results_df.columns and std_col in avg_results_df.columns:
            combined_data[label] = [
                fmt(m, s) for m, s in zip(avg_results_df[mean_col], avg_results_df[std_col])
            ]

    combined_df = pd.DataFrame(combined_data)
    
    combined_df["F1_sort"] = avg_results_df["F1_mean"]
    combined_df = combined_df.sort_values("F1_sort", ascending=False).drop(columns="F1_sort")

    # Display top 10
    print("\nTop 10 setups by F1 Score:")
    print(combined_df.head(10).to_string(index=False))

    # Save to CSV (semicolon separator, comma decimals)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False, sep=';', decimal=',')

    print(f"\nSummary saved to: {output_path}")
