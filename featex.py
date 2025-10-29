import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class FeatureExtractor:
    """
    Returns full coefficient matrices for CNN input.
    Each method returns a 2D or 1D numpy array representing time–frequency data.
    """
    @staticmethod
    def extract_dwt_features(signal, wavelet='db4', level=4, max_length=100):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = np.concatenate([np.ravel(c) for c in coeffs])
        return features[:max_length]
    
    @staticmethod
    def extract_dwt_features_multi_channel(signal, wavelet='db4', level=4, target_length=None):
        """
        Multi-channel version of DWT:
        Each subband (A_n, D_n, D_{n-1}, ...) becomes one channel.

        Returns:
            np.ndarray: shape (n_channels, time_length)
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Pad each subband to same length for CNN input
        max_len = max(len(c) for c in coeffs) if target_length is None else target_length
        padded = []
        for c in coeffs:
            c = np.asarray(c)
            if len(c) < max_len:
                c = np.pad(c, (0, max_len - len(c)), mode='constant')
            else:
                c = c[:max_len]
            padded.append(c)
        
        features = np.stack(padded, axis=0)  # (n_channels, time_length)
        return features

    @staticmethod
    def extract_wpd_features(signal, wavelet='db4', level=3, target_length=None):
        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, order='freq')]

        coeffs = [wp[node].data for node in nodes]

        # Pad or truncate to equal length
        max_len = max(len(c) for c in coeffs) if target_length is None else target_length
        padded = []
        for c in coeffs:
            c = np.asarray(c)
            if len(c) < max_len:
                c = np.pad(c, (0, max_len - len(c)), mode='constant')
            else:
                c = c[:max_len]
            padded.append(c)
        
        features = np.stack(padded, axis=0)  # (n_channels, time_length)
        return features
    
    @staticmethod
    def extract_cwt_features(signal, fs, wavelet='morl',
                          fmin=1, fmax=120,
                          n_freqs=128, n_times=224,
                          log_scale=True, normalize=True):
        # --- Frequency to scale conversion
        freqs = np.geomspace(fmin, fmax, n_freqs)
        central_freq = pywt.central_frequency(wavelet)
        scales = central_freq * fs / freqs

        # --- Continuous wavelet transform
        coef, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coef)

        # --- Optional log compression
        if log_scale:
            power = np.log1p(power)

        # --- Time resampling to fixed grid
        t_src = np.linspace(0, 1, power.shape[1])
        t_dst = np.linspace(0, 1, n_times)
        interp_func = interp1d(t_src, power, kind="linear", axis=1, fill_value="extrapolate")
        scalogram = interp_func(t_dst)

        # --- Optional normalization (per scalogram)
        if normalize:
            mean = scalogram.mean()
            std = scalogram.std() + 1e-8
            scalogram = (scalogram - mean) / std
        return scalogram, freqs


    # ------------------------------------------------------------
    # Not Needed METHODS
    # ------------------------------------------------------------
    @staticmethod
    def extract_time_features(signal):
        """
        Simple statistical descriptors from the time domain.
        """
        signal = np.asarray(signal)
        if len(signal) == 0:
            return np.zeros(8)

        diff = np.diff(signal)
        features = [
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.median(signal),
            np.sum(np.abs(diff)),      # roughness
            np.argmax(signal),         # peak latency
            np.argmin(signal)          # trough latency
        ]
        return np.array(features)
    
    @staticmethod
    def extract_wpd_features_old(signal, wavelet='db4', maxlevel=3):
        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=maxlevel)
        nodes = wp.get_level(maxlevel, order='freq')
        coeffs = [n.data for n in nodes]
        coeffs_norm = [c / (np.std(c) + 1e-6) for c in coeffs]
        maxlen = max(len(c) for c in coeffs_norm)
        coeffs_padded = [np.pad(c, (0, maxlen - len(c))) for c in coeffs_norm]
        return np.stack(coeffs_padded)


    @staticmethod
    def extract_fft_features(signal, fs=1000, n_bands=10, max_freq=200):
        """
        Compute average power in n_bands up to max_freq Hz.
        """
        signal = np.asarray(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1/fs)
        power = np.abs(fft_vals)**2

        band_edges = np.linspace(0, max_freq, n_bands + 1)
        band_powers = []
        for i in range(n_bands):
            mask = (fft_freqs >= band_edges[i]) & (fft_freqs < band_edges[i + 1])
            band_powers.append(np.mean(power[mask]) if np.any(mask) else 0)
        return np.array(band_powers)

    @staticmethod
    def extract_combined_features(signal, wavelet='db4', level=4, fs=1000, max_length=100):
        """
        Combine DWT, time, and frequency-domain features into one vector.
        """
        dwt_feat = FeatureExtractor.extract_dwt_features(signal, wavelet, level, max_length)
        time_feat = FeatureExtractor.extract_time_features(signal)
        fft_feat = FeatureExtractor.extract_fft_features(signal, fs)
        return np.concatenate([dwt_feat, time_feat, fft_feat])

    
    # @staticmethod
    # def extract_cwt_features(signal, fs=2000, wavelet='morl',
    #                          fmin=5, fmax=250, n_freqs=96):
    #     freqs = np.geomspace(fmin, fmax, n_freqs)
    #     scales = pywt.scale2frequency(wavelet, 1.0 / freqs) * fs
    #     coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)
    #     mag = np.abs(coeffs)
    #     mag = (mag - mag.mean(axis=1, keepdims=True)) / (mag.std(axis=1, keepdims=True) + 1e-8)
    #     return mag.astype(np.float32)
    

    @staticmethod
    def extract_stft_features(signal, fs=2000, nperseg=256, noverlap=128):
        from scipy.signal import stft
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        magnitude = 20 * np.log10(np.abs(Zxx) + 1e-6)  # dB scale
        return magnitude
    

    @staticmethod
    def use_raw(signal):
        """Return raw 1D signal as is."""
        return np.asarray(signal, float)
    
    @staticmethod
    def plot_features(feature, feature_type="Raw", signal=None, fs=2000, title=None):
        plt.figure(figsize=(10, 3))

        if feature.ndim == 1:
            plt.plot(feature, color='k', lw=1)
            plt.title(title or f"{feature_type} 1D Features")
            plt.xlabel("Coefficient Index")
            plt.ylabel("Amplitude / Coefficient Value")

        elif feature.ndim == 2:
            plt.imshow(feature, aspect='auto', cmap='viridis', origin='lower')
            plt.title(title or f"{feature_type} 2D Feature Map")
            plt.xlabel("Time (samples)")
            plt.ylabel("Frequency / Scale Index")
            cbar = plt.colorbar()
            cbar.set_label("Magnitude")

        else:
            raise ValueError("Feature must be 1D or 2D array.")

        plt.tight_layout()
        plt.show()

        # show raw signal next to it
        if signal is not None:
            plt.figure(figsize=(10, 3))
            t = np.arange(len(signal)) / fs * 1000
            plt.plot(t, signal, 'k', lw=1)
            plt.title("Original Preprocessed Signal")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.tight_layout()
            plt.show()
