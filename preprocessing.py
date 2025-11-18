from helpers import load_dataset_paths
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import detrend
import os

# get ch1 and ch3 data 
BASE_DIR = "MATLAB_prep/Labelled_VEP_Data"
OUTPUT_DIR = "Preprocessed_VEP_Data"
DEVICES = ["PRIMA", "MP20"]
LABELS = ["BC_Only", "RGC_Only", "BC_and_RGC"]

def preprocess_save_all(BASE_DIR=BASE_DIR, OUTPUT_DIR=OUTPUT_DIR, DEVICES=DEVICES, LABELS=LABELS, normalize=True, tmax=400, delay=0):
    all_paths_raw = load_dataset_paths(base_dir=BASE_DIR, devices=DEVICES, labels=LABELS)
    for device, label_dict in all_paths_raw.items():
        for label, file_list in label_dict.items():
            output_subdir = os.path.join(OUTPUT_DIR, device, label)
            os.makedirs(output_subdir, exist_ok=True)
            
            for file_path in file_list:
                time_preprocessed, signal_preprocessed = preprocess_signal(file_path, tmax=tmax, delay=delay, normalize=normalize)
                
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_subdir, filename)
                
                # Explicit overwrite
                if os.path.exists(output_path):
                    os.remove(output_path)

                df = pd.DataFrame({'Time': time_preprocessed, 'Signal': signal_preprocessed})
                df.to_csv(output_path, index=False)



def preprocess_signal(file, normalize=True, tmax=400, delay=0):
    time, signal = load_signal(file)
    ch3_avg, time_avg = average_two_phases(signal, time)
    #ch3_detrended = detrend_signal(ch3_avg)
    ch3_filtered = filtering(ch3_avg)
    time_clean, ch3_clean = artifact_removal(
        ch3_filtered,
        time_avg,
        file,
        delay=delay,
    )
    time_trimmed, ch3_trimmed = trim(ch3_clean, time_clean, t_min=0, t_max=tmax)

    if normalize:
        ch3_normalized = normalize_signal(ch3_trimmed)
    return time_trimmed, ch3_normalized


def load_signal(file):
    df = pd.read_csv(
        file,
        skiprows=1
    )[['Step 1', 'Chan 3']]

    # Drop the sub-header row
    df = df.drop(index=0).reset_index(drop=True)
    # Convert to numeric
    df = df.apply(pd.to_numeric)

    signal = df['Chan 3'].values
    time = df['Step 1'].values
    return time, signal

def average_two_phases(signal, time):
    N = len(signal) // 2
    # split signal
    sig1 = signal[:N]
    sig2 = signal[N:2*N]
    # average
    signal_avg = (sig1 + sig2) / 2
    time_avg   = time[:N]     
    return signal_avg, time_avg

def detrend_signal(signal):
    return detrend(signal)


def filtering(signal, fs=2000, cutoff=1.5):
    # highpass to remove 2.5 Hz frequency 
    # (can be used INSTEAD of detrending)
    nyq = fs / 2
    b, a = butter(4, cutoff/nyq, btype='highpass')
    filtered = filtfilt(b, a, signal)
    return filtered



def extract_PulseWidth_SignalPower(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    #  FIND PART THAT CONTAINS "ms"
    pulsePart = None
    pulseIndex = None

    for i, p in enumerate(parts):
        if "ms" in p.lower():
            pulsePart = p
            pulseIndex = i
            break

    if pulsePart is None:
        raise ValueError(f"No pulse width ('ms') found in filename: {filename}")
    #  Special case: pulse width split into '0' + '5ms'

    if pulseIndex == 3 and parts[pulseIndex - 1] == '0':
        # combine "0" and "5ms" → "0_5ms"
        pulsePart = f"{parts[pulseIndex - 1]}_{parts[pulseIndex]}"

    # remove "ms"
    pulsePart = pulsePart.replace("ms", "")

    # replace "_" with "." → matches MATLAB strrep
    pulsePart = pulsePart.replace("_", ".")

    # convert to float
    try:
        pulseWidth = float(pulsePart)
    except ValueError:
        raise ValueError(f"Could not parse pulse width from: {pulsePart} (file: {filename})")

    #  Extract irradiance / power
    # last item: e.g., "0.60mWmm2"
    signalPart = parts[-1].replace("mWmm2", "")

    try:
        signalPower = float(signalPart)
    except ValueError:
        raise ValueError(f"Could not parse signal power from: {signalPart} (file: {filename})")

    return pulseWidth, signalPower


def artifact_removal(signal, time, filepath, delay=0):
    pulse_width, _ = extract_PulseWidth_SignalPower(filepath)
    # index after stimulus
    idx_after_pulse = np.where(time > pulse_width)[0][0]
    # 1) baseline shift
    signal = signal - signal[idx_after_pulse]
    # 2) zero out stimulation period (time <= pulse_width)
    signal = np.where(time > pulse_width, signal, 0)
    # 3) shift so stimulus onset = 0 ms, add optional delay
    time = time - pulse_width + delay
    return time, signal

def trim(signal, time, t_min=0, t_max=200):
    mask = (time >= t_min) & (time <= t_max)
    return time[mask], signal[mask]

def normalize_signal(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    return signal
