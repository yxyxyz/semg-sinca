import os
import time
import warnings
import argparse

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils.data_preprocessing import load_and_seg_data, CNNFeatureDataset

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FeatureDataset(Dataset):
    """Dataset for STFT features."""
    def __init__(self, stft_features, labels):
        self.stft_features = stft_features.astype(np.float32)
        self.labels = labels

    def __getitem__(self, index):
        # Convert to tensor and permute to (channels, freq_bins, time_steps)
        features = torch.tensor(self.stft_features[index]).permute(1, 0, 2)
        return features, torch.tensor(int(self.labels[index]))

    def __len__(self):
        return len(self.stft_features)


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def process_subject(subject, method, data_dir='data'):
    """Process a single subject."""
    print(f"Processing subject {subject} with method {method}")

    try:
        dataset_path = os.path.join(data_dir, f"DB2_s{subject}", f"S{subject}_E1_A1.mat")

        if method == 'pca_svm':
            window_size, step_size = 600, 20
        else:
            window_size, step_size = 600, 20

        X_train, y_train = load_and_seg_data(dataset_path, [1, 3, 4, 6], window_size, step_size)
        print(f"Training data shape: {X_train.shape}")
        X_test, y_test = load_and_seg_data(dataset_path, [2, 5], window_size, step_size)

        X_train_scaled = np.zeros_like(X_train)
        X_test_scaled = np.zeros_like(X_test)
        for i in range(X_train.shape[2]):
            mean_train = np.mean(X_train[:, :, i])
            std_train = np.std(X_train[:, :, i])
            X_train_scaled[:, :, i] = (X_train[:, :, i] - mean_train) / (std_train + 1e-8)
            X_test_scaled[:, :, i] = (X_test[:, :, i] - mean_train) / (std_train + 1e-8)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"No data available for subject {subject}")
            return 0.0

    except Exception as e:
        print(f"Error loading data for subject {subject}: {e}")
        return 0.0

    if method == 'pca_svm':
        X_train_feat = []
        for window in X_train_scaled:
            spectrogram = _compute_spectrogram(window)
            if spectrogram is not None:
                X_train_feat.append(spectrogram)

        X_test_feat = []
        for window in X_test_scaled:
            spectrogram = _compute_spectrogram(window)
            if spectrogram is not None:
                X_test_feat.append(spectrogram)

        X_train_feat = np.array(X_train_feat)
        X_test_feat = np.array(X_test_feat)

        y_train_feat = y_train[:len(X_train_feat)]
        y_test_feat = y_test[:len(X_test_feat)]

        accuracy = pca_svm_method(X_train_feat, y_train_feat, X_test_feat, y_test_feat)

    else:  # cnn_lstm
        X_train_feat = []
        for window in X_train_scaled:
            spectrogram = _compute_time_step_spectrogram(window, nperseg=128, noverlap=64)
            if spectrogram is not None:
                X_train_feat.append(spectrogram)

        X_test_feat = []
        for window in X_test_scaled:
            spectrogram = _compute_time_step_spectrogram(window, nperseg=128, noverlap=64)
            if spectrogram is not None:
                X_test_feat.append(spectrogram)

        X_train_feat = np.array(X_train_feat)
        X_test_feat = np.array(X_test_feat)

        y_train_feat = y_train[:len(X_train_feat)]
        y_test_feat = y_test[:len(X_test_feat)]
        print(X_train_feat.shape)

        accuracy = cnn_lstm_method(X_train_feat, y_train_feat, X_test_feat, y_test_feat)

    print(f"Subject {subject} accuracy: {accuracy:.4f}")
    return accuracy


def _compute_spectrogram(segment, fs=2000, nperseg=256, noverlap=184):
    """Compute spectrogram for PCA+SVM method."""
    try:
        n_channels = segment.shape[1]
        channel_spectrograms = []

        for ch in range(n_channels):
            f, t, Sxx = signal.spectrogram(
                segment[:, ch], fs=fs,
                window=signal.windows.hann(nperseg, sym=False),
                nperseg=nperseg, noverlap=noverlap
            )
            Sxx = Sxx[:95, :]
            channel_spectrograms.append(Sxx.flatten())

        return np.concatenate(channel_spectrograms)
    except Exception as e:
        print(f"Error computing spectrogram: {e}")
        return None


def _compute_time_step_spectrogram(segment, fs=2000, nperseg=128, noverlap=128):
    """Compute time-step spectrogram for CNN-LSTM method."""
    try:
        n_channels = segment.shape[1]
        channel_spectrograms = []

        for ch in range(n_channels):
            f, t, Sxx = signal.spectrogram(
                segment[:, ch], fs=fs,
                window=signal.windows.hann(nperseg, sym=False),
                nperseg=nperseg, noverlap=noverlap
            )
            Sxx = np.abs(Sxx)
            channel_spectrograms.append(Sxx)

        return np.stack(channel_spectrograms, axis=-1)
    except Exception as e:
        print(f"Error computing spectrogram: {e}")
        return None


# =============================================================================
# PCA + SVM method
# =============================================================================

def pca_svm_method(X_train, y_train, X_test, y_test):
    """PCA + SVM classification."""
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_normalized)
    X_test_pca = pca.transform(X_test_normalized)

    param_grid = {
        'C': [2 ** i for i in range(-2, 15)],
        'gamma': [2 ** i for i in range(-12, 8)]
    }

    svm = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm, param_grid, cv=4, n_jobs=-1, verbose=0)
    grid_search.fit(X_train_pca, y_train)

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# =============================================================================
# CNN-LSTM model (PyTorch)
# =============================================================================

class CNNLSTM(nn.Module):
    """CNN-LSTM model as described in the paper."""
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()

        self.time_steps = 8
        self.channels = 12
        self.freq_bins = 65

        # CNN part
        self.conv1 = nn.Conv1d(self.channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(64)

        # LSTM part
        self.lstm1 = nn.LSTM(
            input_size=64 * self.freq_bins,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.lstm2 = nn.LSTM(
            input_size=1024,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape: (batch * time_steps, channels, freq_bins)
        x = x.view(batch_size * self.time_steps, self.channels, self.freq_bins)

        # CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        # Reshape: (batch, time_steps, features)
        x = x.view(batch_size, self.time_steps, -1)

        # LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Take last time step
        x = x[:, -1, :]

        # Fully connected
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)

        return x


def train_cnn_lstm(model, train_loader, val_loader, num_epochs=200,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train CNN-LSTM model with learning rate scheduling."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.95, weight_decay=0.005)

    milestones = [20, 40, 80, 100, 140]
    gamma_values = [0.2 / 0.5, 0.1 / 0.2, 0.05 / 0.1, 0.02 / 0.05, 0.01 / 0.02]

    def lr_lambda(epoch):
        if epoch < milestones[0]:
            return 1.0
        elif epoch < milestones[1]:
            return gamma_values[0]
        elif epoch < milestones[2]:
            return gamma_values[0] * gamma_values[1]
        elif epoch < milestones[3]:
            return gamma_values[0] * gamma_values[1] * gamma_values[2]
        elif epoch < milestones[4]:
            return gamma_values[0] * gamma_values[1] * gamma_values[2] * gamma_values[3]
        else:
            return gamma_values[0] * gamma_values[1] * gamma_values[2] * gamma_values[3] * gamma_values[4]

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        if epoch in milestones:
            print(f"Epoch {epoch + 1}: Learning rate changed from {current_lr:.4f} to {new_lr:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {new_lr:.4f}')

    return best_acc / 100


def cnn_lstm_method(X_train, y_train, X_test, y_test, num_classes=17):
    """Run CNN-LSTM method."""
    mean_stft = np.mean(X_train, axis=(0, 1, 2), keepdims=True)
    std_stft = np.std(X_train, axis=(0, 1, 2), keepdims=True)
    X_train_normalized = (X_train - mean_stft) / (std_stft + 1e-8)
    X_test_normalized = (X_test - mean_stft) / (std_stft + 1e-8)

    train_dataset = FeatureDataset(X_train_normalized, y_train)
    test_dataset = FeatureDataset(X_test_normalized, y_test)

    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    model = CNNLSTM(num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    accuracy = train_cnn_lstm(model, train_loader, test_loader, device=device)
    return accuracy


# =============================================================================
# Main function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Surface EMG Gesture Recognition')
    parser.add_argument('--method', type=str, choices=['pca_svm', 'cnn_lstm'],
                        required=True, help='Method to use: pca_svm or cnn_lstm')
    parser.add_argument('--data_dir', type=str, default='./data/DB2', help='Path to data directory')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 41)),
                        help='List of subjects to process (default: 1-40)')
    parser.add_argument('--output', type=str, default='c_results.csv', help='Output CSV file')

    args = parser.parse_args()

    results = []

    for subject in args.subjects:
        try:
            start_time = time.time()
            accuracy = process_subject(subject, args.method, args.data_dir)
            end_time = time.time()

            results.append({
                'subject': subject,
                'method': args.method,
                'accuracy': accuracy,
                'time_seconds': end_time - start_time
            })

            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)

        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            continue

    print("\n=== Overall Results ===")
    df = pd.DataFrame(results)
    print(df)
    print(f"\nAverage accuracy: {df['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()