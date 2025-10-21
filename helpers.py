
import numpy as np
import matplotlib.pyplot as plt
import os
import re 
import pandas as pd


BASE_DIR = "Labelled_VEP_Data"
DEVICES = ["PRIMA", "MP20"]
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

def load_vep_csv(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract column names and units
    if len(lines) < 3:
        raise ValueError(f"File {file_path} too short or not a valid VEP CSV")

    column_names = [x.strip() for x in lines[1].split(",")]
    units = [x.strip() for x in lines[2].split(",")]

    # Load data (skip first 3 lines)
    df = pd.read_csv(file_path, skiprows=3, header=None)
    df.columns = column_names

    # Remove empty columns (some CSVs have trailing commas)
    df = df.dropna(axis=1, how="all")

    return df, units


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
