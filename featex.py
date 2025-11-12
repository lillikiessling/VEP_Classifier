import pywt
import numpy as np


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
    