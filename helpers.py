
import numpy as np
import os
import pandas as pd


BASE_DIR = "Preprocessed_VEP_Data"
DEVICES = ["PRIMA"]
LABELS = ["BC_Only", "RGC_Only", "BC_and_RGC"]
fs = 2000


def load_preprocessed_signal(file):
    df = pd.read_csv(file)

    # No skiprows, no sub-header
    df = df[['Time', 'Signal']]

    signal = df['Signal'].values
    time = df['Time'].values
    return time, signal


def process_file(filepath, delay=0, t_min=0, t_max=200, normalize=True):
    # 1) Extract pulsewidth from summary file
    # Get device and category from filepath Assuming structure: BASE_DIR / DEVICE / CATEGORY / filename.csv
    parts = os.path.normpath(filepath).split(os.sep)
    device = parts[-3]
    category = parts[-2]

    # Load summary file for this device & category
    summary_path = os.path.join(BASE_DIR, device, category, f"SNR_summary_{category}.csv")
    summary_df = pd.read_csv(summary_path)

    # Find matching file row in summary
    summary_df["FileName"] = summary_df["FileName"].astype(str).str.strip()
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    match = summary_df[summary_df["FileName"] == file_name]
    if match.empty:
        raise ValueError(f"No matching file found in summary for {file_name} ({device}/{category})")
    # Extract pulse width
    pulse_width = float(match["PulseWidth_ms"].iloc[0])

    # 2) Load and process raw data
    df = pd.read_csv(filepath, header=None, names=["time_ms", "ch3", "ch1"])
    idx_after_pulse = df.index[df["time_ms"] > pulse_width][0]

    # Align ch1 and ch3 so that they are zero after pulse
    df["ch1"] = df["ch1"] - df.loc[idx_after_pulse, "ch1"]
    df["ch3"] = df["ch3"] - df.loc[idx_after_pulse, "ch3"]
    df["ch3"] = df["ch3"].where(df["time_ms"] > pulse_width, 0)
    df["time_ms"] = df["time_ms"] - pulse_width + delay

    # Trim to [t_min, t_max]
    df_sliced = df[(df["time_ms"] >= t_min) & (df["time_ms"] <= t_max)].copy()
    time = df_sliced["time_ms"].values
    signal = df_sliced["ch3"].values

    # Normalize (zero mean, unit std)
    if normalize:
        signal = (signal - np.mean(signal)) / np.std(signal)
    return time, signal


def compute_average_signal(df_list, t_min=0, t_max=200):
    """
    Computes the average VEP signal (Ch1) from a list of PROCESSED DataFrames.
    """
    all_times, all_signals = [], []

    for df in df_list: 
        time, signal = process_file(df, t_min=t_min, t_max=t_max, normalize=True)
        if len(signal) > 0:
            all_times.append(time)
            all_signals.append(signal)
    
    avg_t = all_times[0]  # all 'time' vectors are identical
    
    signals_matrix = np.stack(all_signals, axis=1) 
    avg_sig = np.mean(signals_matrix, axis=1)
    return avg_t, avg_sig



def load_dataset_paths(base_dir=BASE_DIR, devices=DEVICES, labels=LABELS, ext=".csv"):
    """
    Returns a nested dictionary with all file paths per device and label.
    Example: paths["PRIMA"]["BC_Only"] list of CSV file paths
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


def compute_average_signal(file_list, t_min=0, t_max=200, normalize=True, delay=0):
    all_times = []
    all_signals = []

    for filepath in file_list:
        time, signal = process_file(filepath, delay=delay, t_min=t_min, t_max=t_max, normalize=normalize)
        if len(signal) > 0:
            all_times.append(time)
            all_signals.append(signal)

    # Assumes all time vectors are identical
    avg_time = all_times[0]
    signals_matrix = np.stack(all_signals, axis=1)
    avg_signal = np.mean(signals_matrix, axis=1)
    return avg_time, avg_signal



# TODO: Needs to be rewritten
def summarize_results_and_save(average_results, model_name = "Model"):
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
