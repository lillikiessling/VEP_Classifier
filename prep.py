import numpy as np
from scipy.signal import find_peaks, detrend
import pywt
import re
from helpers import parse_vep_filename, load_vep_csv
import os
from scipy.signal import welch
from scipy.stats import entropy

class Preprocessor:
    """
    Preprocessing class for biosignals (e.g., retinal or cortical recordings).
    Includes detrending, normalization, frequency filtering, and artifact removal.
    """

    def __init__(self, fs):
        self.fs = fs  # Sampling frequency in Hz

    # ---------- Basic signal operations ----------

    @staticmethod
    def convert_to_uV(signal, scale_factor=0.001):
        """Convert signal from nV to µV (default 1 nV = 0.001 µV)."""
        return np.asarray(signal) * scale_factor

    @staticmethod
    def average_signal_halves(signal):
        """Average two equal-length halves of the signal (like MATLAB phase averaging)."""
        even_len = len(signal) - (len(signal) % 2)
        signal = signal[:even_len]
        N = even_len // 2
        return (signal[:N] + signal[N:]) / 2

    @staticmethod
    def normalize_signal(signal):
        """Normalize signal to max absolute value = 1."""
        max_val = np.max(np.abs(signal))
        return signal if max_val == 0 else signal / max_val

    @staticmethod
    def detrend_signal(signal):
        """Remove slow drifts from the signal."""
        return detrend(signal)

    # ---------- Frequency filtering ----------

    def frequency_domain_filter(self, signal, filt_freq=32, harmonics=20, bandwidth=1):
        """
        Zero out narrow bands around specific harmonics in the frequency domain.
        """
        n = len(signal)
        freq = np.fft.rfftfreq(n, d=1/self.fs)
        signal_fft = np.fft.rfft(signal)

        mask = np.ones_like(freq, dtype=bool)
        for h in range(1, harmonics + 1):
            center = filt_freq * h
            mask &= (np.abs(freq - center) > bandwidth)

        filtered_fft = signal_fft * mask
        return np.fft.irfft(filtered_fft, n=n)

    # ---------- Artifact removal ----------

    @staticmethod
    def artifact_removal(ch1, ch3, pulse_dur_samples, baseline_idx=20, local_only=True):
        """
        Remove stimulation artifacts by scaling ch3 relative to ch1.
        """
        ch1, ch3 = np.asarray(ch1, float), np.asarray(ch3, float)
        if len(ch1) != len(ch3):
            raise ValueError("ch1 and ch3 must have the same length")

        if len(ch3) <= baseline_idx:
            raise ValueError("Signal too short for artifact removal (needs ≥ baseline_idx + 1 samples).")

        ch1 -= ch1[baseline_idx]
        ch3 -= ch3[baseline_idx]

        max_range_end = min(int(baseline_idx + pulse_dur_samples * 2), len(ch1))
        max_range = slice(0, max_range_end)

        ind = np.argmax(np.abs(ch1[max_range]))
        ch1_max = np.abs(ch1[max_range][ind])
        ch3_max = ch3[max_range][ind]

        denom = (ch3_max - ch3[baseline_idx])
        scale = (ch1_max - ch1[baseline_idx]) / denom if not np.isclose(denom, 0) else 1.0

        if local_only:
            end_idx = int(baseline_idx + pulse_dur_samples * 2)
            ch3_segment = ch3[baseline_idx:end_idx] - ch1[baseline_idx:end_idx] / scale
            ch3_noarti = np.concatenate([ch3[:baseline_idx], ch3_segment, ch3[end_idx:]])
        else:
            ch3_noarti = ch3 - ch1 / scale

        return ch3_noarti

    # ---------- Full preprocessing pipeline ----------

    def preprocess(self, ch1, ch3, pulse_dur_ms, acuity=False, normalize=False, trim_ms=None):
        """
        Full preprocessing pipeline:
        1. Convert to µV
        2. Phase-average
        3. Optional frequency-domain filtering (acuity=True)
        4. Artifact removal
        5. Detrend
        6. Optional normalization
        """
        ch1 = self.convert_to_uV(ch1)
        ch3 = self.convert_to_uV(ch3)

        ch1 = self.average_signal_halves(ch1)
        ch3 = self.average_signal_halves(ch3)

        if trim_ms is not None:
            n_samples = int((trim_ms / 1000) * self.fs)
            ch1 = ch1[:-n_samples] if n_samples < len(ch1) else ch1
            ch3 = ch3[:-n_samples] if n_samples < len(ch3) else ch3

        if acuity:
            ch3 = self.frequency_domain_filter(ch3)

        pulse_dur_samples = int((pulse_dur_ms / 1000) * self.fs)
        ch3 = self.artifact_removal(ch1, ch3, pulse_dur_samples, local_only=True)

        ch3 = self.detrend_signal(ch3)

        if normalize:
            ch3 = self.normalize_signal(ch3)

        return ch3

