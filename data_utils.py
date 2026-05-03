import random
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy
from scipy.signal import stft

def load_and_seg_data(file_path, rep_set, window_size, step_size):
    data = scipy.io.loadmat(file_path)
    emg = data['emg']  # shape (n_samples, n_channels)
    restimulus = data['restimulus'].flatten()  # flatten to 1D array
    repetition = data['repetition'].flatten()  # flatten to 1D array

    inputs = []
    labels = []
    for rep in rep_set:
        # Create mask for current repetition
        rep_mask = (repetition == rep)
        rep_indices = np.where(rep_mask)[0]

        if not rep_indices.size:
            continue
        for resti in range(1, 18):
            # Create mask for current restimulus within repetition
            resti_mask = (restimulus[rep_indices] == resti)
            resti_indices = rep_indices[resti_mask]

            if len(resti_indices) < window_size:
                del resti_mask, resti_indices
                continue

            # Calculate number of possible windows
            num_windows = (len(resti_indices) - window_size) // step_size

            for step in range(num_windows):
                start_idx = step * step_size
                end_idx = start_idx + window_size
                # Get absolute start and end positions
                start_pos = resti_indices[start_idx]
                end_pos = resti_indices[end_idx - 1]  # -1 for inclusive index
                # Verify start and end have same repetition and restimulus
                if (repetition[start_pos] == repetition[end_pos] and
                        restimulus[start_pos] == restimulus[end_pos]):
                    # Extract window and add to results
                    window_emg = emg[resti_indices[start_idx:start_idx + window_size]]
                    inputs.append(window_emg)
                    labels.append(resti)
            # Clean up restimulus-specific variables
            del resti_mask, resti_indices
        # Clean up repetition-specific variables
        del rep_mask, rep_indices
    # Convert to arrays and adjust labels
    inputs = np.array(inputs)
    labels = np.array(labels) - 1  # make labels 0-indexed
    return inputs, labels

# --- Custom Dataset Classes ---
class STFTFeatureDataset(Dataset):
    def __init__(self, raw_data, target_stft_features):
        self.raw_data = raw_data.astype(np.float32)
        self.target_stft_features = target_stft_features.astype(np.float32)

    def __getitem__(self, index):
        return torch.tensor(self.raw_data[index]), torch.tensor(self.target_stft_features[index])

    def __len__(self):
        return len(self.raw_data)


class CNNFeatureDataset(Dataset):
    def __init__(self, stft_features, labels):
        self.stft_features = stft_features.astype(np.float32)
        self.labels = labels

    def __getitem__(self, index):
        # Convert to tensor and permute to (channels, freq_bins, time_steps)
        features = torch.tensor(self.stft_features[index]).permute(2, 0, 1)
        return features, torch.tensor(int(self.labels[index]))

    def __len__(self):
        return len(self.stft_features)


class sEMGDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def extract_stft_features(data, nperseg=48, fs=2000):
    """
    Computes STFT features (magnitude) from raw sEMG data.

    Args:
        data (np.ndarray): Raw sEMG data with shape (batch_size, sequence_length, channels).

    Returns:
        np.ndarray: STFT magnitudes with shape (batch_size, num_frequency_bins, num_time_steps, channels).
                    num_frequency_bins = nperseg // 2 + 1
                    num_time_steps depends on sequence_length, nperseg, and noverlap.
        np.ndarray: The frequency bins.
        np.ndarray: The time steps.
    """
    batch_size, sequence_length, channels = data.shape

    # Define STFT parameters
    #nperseg = 64 #64 # Window size
    noverlap = nperseg // 2 # 50% overlap
    #fs = 2000 # Assuming sampling frequency of 1000 Hz

    all_stft_magnitudes = []
    freqs = None
    times = None

    for i in range(batch_size):
        sample_stft_magnitudes = []
        for j in range(channels):
            # Compute STFT for each channel
            f, t, Zxx = stft(data[i, :, j], fs=fs, nperseg=nperseg, noverlap=noverlap)

            # Add a print statement to check the shape of Zxx for the first sample and channel
            # if i == 0 and j == 0:
            #    print(f"Shape of Zxx from scipy.signal.stft for first sample and channel: {Zxx.shape}")
            #    print(f"Time steps calculated by scipy: {t.shape[0]}")


            # Use the magnitude of the STFT
            magnitude = np.abs(Zxx) # shape: (num_frequency_bins, num_time_steps)

            sample_stft_magnitudes.append(magnitude)

            if freqs is None:
                freqs = f
                times = t

        # Stack magnitudes for all channels in the current sample
        # Resulting shape: (num_frequency_bins, num_time_steps, channels)
        sample_stft_magnitudes = np.stack(sample_stft_magnitudes, axis=-1)
        all_stft_magnitudes.append(sample_stft_magnitudes)

    # Stack magnitudes for all samples in the batch
    # Resulting shape: (batch_size, num_frequency_bins, num_time_steps, channels)
    all_stft_magnitudes = np.stack(all_stft_magnitudes, axis=0)

    return all_stft_magnitudes, freqs, times

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
