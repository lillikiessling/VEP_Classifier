
import numpy as np
import matplotlib.pyplot as plt
import os
import re 
import pandas as pd
from tqdm import tqdm
import scipy.signal as sp


BASE_DIR = "Preprocessed_VEP_Data"
DEVICES = ["PRIMA"]
LABELS = ["BC_Only", "RGC_Only", "BC_and_RGC"]
fs = 2000

def parse_filename(fname):
    """
    Handles patterns like:
    PRIMA100_9_10ms_0.60mWmm2.csv
    PRIMA100_7_1ms_1.53mWmm2_2.csv
    Pattern: Device_animal_pulsewidth_irradiance(_rep).csv
    """
    base = os.path.basename(fname)
    name = base[:-4] if base.lower().endswith(".csv") else base
    parts = name.split("_")

    # Remove trailing numeric repetition indicator (e.g. '_2')
    if parts[-1].isdigit():
        parts = parts[:-1]

    # Irradiance is now last
    irr_token = parts[-1]
    if not irr_token.endswith("mWmm2"):
        return None, None
    irr_str = irr_token.replace("mWmm2", "").replace("_", ".")
    try:
        irradiance = float(irr_str)
    except ValueError:
        return None, None

    # Pulse width is last token ending with 'ms'
    pulse_token = None
    for tok in reversed(parts[:-1]):
        if tok.endswith("ms"):
            pulse_token = tok[:-2]
            break
    if pulse_token is None:
        return None, None

    pulse_str = pulse_token.replace("_", ".")
    try:
        pulse_ms = float(pulse_str)
    except ValueError:
        return None, None

    return pulse_ms, irradiance

def bandpass_filter(signal, fs=2000, lowcut=1, highcut=30, order=4): # currently not used 
    """
    Apply zero-phase Butterworth band-pass filter to 1D signal.
    fs      : sampling rate [Hz]
    lowcut  : low cutoff frequency [Hz]
    highcut : high cutoff frequency [Hz]
    order   : filter order
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.butter(order, [low, high], btype='band')
    filtered = sp.filtfilt(b, a, signal)
    return filtered


def preprocess_signal(time, signal, t_min, t_max, normalize=True, filter=True):
    """
    Trim signal to [t_min, t_max] ms and optionally z-normalize it.
    """
    mask = (time >= t_min) & (time <= t_max)
    time_trimmed = time[mask]
    signal_trimmed = signal[mask]

    # Apply bandpass filter
    # if filter:
    #     signal_trimmed = bandpass_filter(signal_trimmed, fs=fs, lowcut=0.5, highcut=20, order=4)

    # if normalize:
    #     signal_trimmed = (signal_trimmed - np.mean(signal_trimmed)) / np.std(signal_trimmed)
    #     #signal_trimmed = np.clip(signal_trimmed, -3, 3)  
    
    # peak to peak normalization
    if normalize:
        peak_to_peak = np.max(signal_trimmed) - np.min(signal_trimmed)
        signal_trimmed = signal_trimmed / peak_to_peak
        #signal_trimmed = signal_trimmed / np.max(signal_trimmed)
    return time_trimmed, signal_trimmed

def load_vep_csv(file_path, t_min=10, t_max=125, normalize=True):
    """
    Loads a VEP CSV file, skipping non-numeric header lines automatically.
    Trims to [t_min, t_max] and optionally normalizes the signal.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Keep only numeric rows (two comma-separated numbers)
    numeric_lines = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            float(parts[0]); float(parts[1])
            numeric_lines.append(line)
        except ValueError:
            continue

    if not numeric_lines:
        raise ValueError(f"No numeric data found in {file_path}")

    # Load numeric data only
    data = np.loadtxt(numeric_lines, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 2)

    time = data[:, 0]
    signal = data[:, 1]
    time, signal = preprocess_signal(time, signal, t_min=t_min, t_max=t_max, normalize=normalize)
    return time, signal


def load_vep_csv_old(file_path, t_min=10, t_max=125, normalize=True):
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
