import time
import torch
import numpy as np
import os
import random

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import multiprocessing
from data_utils import load_and_seg_data, extract_stft_features, sEMGDataset
from model import ChannelAttentionCNN, SINCA_s, SINCA_xs, SINCA_xxs


def train_model(feature_model, model, train_loader, val_loader, optimizer, criterion,
                num_epochs, save_path, writer, csv_writer, stage_name='training',
                device=None):
    """
    Generic training function that handles both single models and combined models

    Args:
        model: Main model (classifier or feature extractor)
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of training epochs
        save_path: Path to save the model
        writer: TensorBoard writer
        csv_writer: CSV writer
        stage_name: Name of the training stage
        patience: Patience for early stopping
        device: Training device
        feature_model: Feature extraction model (for combined models)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move models to device
    model.to(device)
    if feature_model is not None:
        feature_model.to(device)
    patience = 30
    is_combined_model = feature_model is not None
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # Set models to training mode
        model.train()
        if is_combined_model:
            feature_model.train()

        train_losses, train_accs = [], []

        for batch_data in train_loader:
            # Move data to device
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            if is_combined_model:
                # Combined model: extract features first, then classify
                features = feature_model(inputs)
                outputs = model(features)
            else:
                # Single model: process inputs directly
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(acc.item())

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accs)

        writer.add_scalar(f'{stage_name}/Train_Loss', epoch_train_loss, epoch)
        writer.add_scalar(f'{stage_name}/Train_Acc', epoch_train_acc, epoch)

        # Validation phase
        if val_loader is not None:
            model.eval()
            if is_combined_model:
                feature_model.eval()

            val_losses, val_accs = [], []
            with torch.no_grad():
                for batch_data in val_loader:
                    # Move data to device
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    if is_combined_model:
                        features = feature_model(inputs)
                        outputs = model(features)
                    else:
                        outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1)
                    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
                    val_losses.append(loss.item())
                    val_accs.append(acc.item())

            epoch_val_loss = np.mean(val_losses)
            epoch_val_acc = np.mean(val_accs)

            writer.add_scalar(f'{stage_name}/Val_Loss', epoch_val_loss, epoch)
            writer.add_scalar(f'{stage_name}/Val_Acc', epoch_val_acc, epoch)

            # Log to CSV
            csv_writer.writerow([
                epoch + 1, stage_name,
                epoch_train_loss, epoch_val_loss,
                epoch_train_acc, epoch_val_acc
            ])

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, "
                f"Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, "
                f"Val Acc: {epoch_val_acc:.4f}")

            # Early stopping check
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                epochs_no_improve = 0
                # Save model
                if is_combined_model:
                    torch.save({
                        'stft_imit_net': feature_model.state_dict(),
                        'classifier': model.state_dict(),
                    }, save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
            else:
                epochs_no_improve += 1
        else:
            # No validation set
            csv_writer.writerow([
                epoch + 1, stage_name,
                epoch_train_loss, None,
                epoch_train_acc, None
            ])

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, "
                f"Train Acc: {epoch_train_acc:.4f}")

            # Early stopping check (based on training accuracy)
            if epoch_train_acc > best_acc:
                best_acc = epoch_train_acc
                epochs_no_improve = 0
                # Save model
                if is_combined_model:
                    torch.save({
                        'stft_imit_net': feature_model.state_dict(),
                        'classifier': model.state_dict(),
                    }, save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
            else:
                epochs_no_improve += 1

        # Check for early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        writer.flush()
        print(f"Epoch time: {time.time() - start_time:.2f}s")

    return best_acc

class Runner:
    def __init__(self, config):
        """
        Experiment runner

        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.set_seeds(config.get('seed', 42))
        self.device = self.get_device()

        # Create output directory
        self.run_dir = config['run_dir']
        os.makedirs(self.run_dir, exist_ok=True)

        # Initialize components
        self.model = self.get_model(
            config['model_type'],
            config.get('num_channels', 12),
            config.get('num_classes', 17)
        ).to(self.device)

        self.optimizer = config['optimizer']['class'](
            self.model.parameters(),
            **config['optimizer']['params']
        )
        self.criterion = config['criterion']()
        self.num_epochs = config.get('num_epochs', 30)
        # Prepare data
        self.prepare_data()

        # Initialize training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'pretrain_loss': [], 'pretrain_acc': [],
            'finetune_loss': [], 'finetune_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': None, 'test_acc': None
        }

        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "logs")) if config.get('use_tensorboard',
                                                                                              True) else None

    def set_seeds(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def get_device(self):
        """Get training device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self):
        """Prepare data loaders"""
        # Get configuration parameters
        subject = self.config.get('subject', 1)
        data_dir = self.config['data_dir']
        model_type = self.config['model_type']
        pretrain_mode = self.config.get('pretrain', False)

        # Create dataset
        dataset_path = os.path.join(data_dir, f"DB2_s{subject}", f"S{subject}_E1_A1.mat")
        X_train_raw, y_train = load_and_seg_data(dataset_path, [1, 3, 4, 6], window_size=600, step_size=20)
        X_test_raw, y_test = load_and_seg_data(dataset_path, [2, 5], window_size=600, step_size=20)
        # print(len(y_test))
        # Standardize raw data - use training data to compute standardization parameters
        mean_train = np.mean(X_train_raw, axis=(0, 1), keepdims=True)
        std_train = np.std(X_train_raw, axis=(0, 1), keepdims=True) + 1e-8

        X_train_scaled = (X_train_raw - mean_train) / std_train
        X_test_scaled = (X_test_raw - mean_train) / std_train

        train_features = X_train_scaled
        test_features = X_test_scaled

        if model_type == "stft+cacnn":
            # Extract STFT features
            X_train_stft, _, _ = extract_stft_features(X_train_scaled)
            X_test_stft, _, _ = extract_stft_features(X_test_scaled)

            # Standardize STFT features
            mean_stft = np.mean(X_train_stft, axis=(0, 1, 2), keepdims=True)
            std_stft = np.std(X_train_stft, axis=(0, 1, 2), keepdims=True) + 1e-8
            train_features = (X_train_stft - mean_stft) / std_stft
            test_features = (X_test_stft - mean_stft) / std_stft
            train_features = np.transpose(train_features, (0, 3, 1, 2))
            test_features = np.transpose(test_features, (0, 3, 1, 2))
        # Create datasets
        train_dataset = sEMGDataset(train_features, y_train)
        test_dataset = sEMGDataset(test_features, y_test)

        # Dataset splitting
        val_size = int(len(train_dataset) * self.config.get('val_split', 0.2))
        train_size = len(train_dataset) - val_size
        self.num_epochs = train_size // self.config['batch_size']
#       if self.num_epochs < 30:
#            self.num_epochs = 30
        num_workers = self.config.get('num_workers', min(4, multiprocessing.cpu_count()))
        train_data = train_dataset
        self.val_loader = None
        if val_size != 0:
            # train_size1 = int(train_size * 0.8)
            # train_size2 = train_size - train_size1
            train_data, val_data = random_split(train_dataset, [train_size, val_size])
            self.val_loader = DataLoader(
                val_data,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        # self.val_loader = None
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_model(self, model_type, num_channels=12, num_classes=17):
        """Get model based on model type"""

        if model_type == "stft+cacnn":
            return ChannelAttentionCNN(num_classes=17, input_channels=12)
        elif model_type == "sinca_xxs":
            return SINCA_xxs(
                num_classes=num_classes,
                input_channels=num_channels
            )
        elif model_type == "sinca_xs":
            return SINCA_xs(
                num_classes=num_classes,
                input_channels=num_channels
            )
        elif model_type == "sinca_s":
            return SINCA_s(
                num_classes=num_classes,
                input_channels=num_channels
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_epoch(self, loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = len(loader)

        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # More efficient memory usage
            loss.backward()

            # Gradient clipping
            if self.config.get('clip_grad') is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad'))

            self.optimizer.step()

            # Record metrics
            total_loss += loss.item()
            if self.config.get('is_classifier', True):
                _, preds = torch.max(outputs, dim=1)
                correct = (preds == targets)
                total_acc += correct.sum().item() / len(targets)

        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches if self.config.get('is_classifier', True) else None

        return avg_loss, avg_acc

    def evaluate(self, loader, desc="Evaluating"):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = len(loader)

        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                if self.config.get('is_classifier', True):
                    _, preds = torch.max(outputs, dim=1)
                    correct = (preds == targets)
                    total_acc += correct.sum().item() / len(targets)

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches if self.config.get('is_classifier', True) else None

        return avg_loss, avg_acc

    def log_epoch(self, epoch, phase, loss, acc, val_loss=None, val_acc=None):
        """Log epoch metrics"""
        # Console output
        info = f"{phase} Epoch {epoch + 1} - Loss: {loss:.4f}"
        if self.config.get('is_classifier', True):
            info += f", Acc: {acc:.4f}"

        if val_loss is not None:
            info += f", Val Loss: {val_loss:.4f}"
            if self.config.get('is_classifier', True):
                info += f", Val Acc: {val_acc:.4f}"

        print(info)

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
            if self.config.get('is_classifier', True):
                self.writer.add_scalar(f'Accuracy/{phase}', acc, epoch)

            if val_loss is not None:
                self.writer.add_scalar(f'Loss/{phase}_Val', val_loss, epoch)
                if self.config.get('is_classifier', True):
                    self.writer.add_scalar(f'Accuracy/{phase}_Val', val_acc, epoch)

    def run_training_phase(self, train_loader, num_epochs, phase_name, model_save_path,
                           early_stopping_patience=None):
        """Run a training phase (pretraining or fine-tuning)"""
        best_val_acc = 0.0
        epochs_no_improve = 0
        epochs = num_epochs
        if self.config['model_type'] == "time_domain+cnn":
            epochs = 10
        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history[f'{phase_name}_loss'].append(train_loss)
            if self.config.get('is_classifier', True):
                self.history[f'{phase_name}_acc'].append(train_acc)

            # Validation
            val_loss, val_acc = None, None
            if self.val_loader:
                val_loss, val_acc = self.evaluate(self.val_loader, "Validating")
                self.history['val_loss'].append(val_loss)
                if self.config.get('is_classifier', True):
                    self.history['val_acc'].append(val_acc)

            # Logging
            self.log_epoch(epoch, phase_name, train_loss, train_acc, val_loss, val_acc)

            # Save best model
            current_acc = val_acc if val_acc is not None else train_acc
            if current_acc is not None and current_acc > best_val_acc:
                best_val_acc = current_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Saved best model to {model_save_path}")
            else:
                epochs_no_improve += 1

            # Early stopping check
            if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                print(f"{phase_name} early stopping triggered at epoch {epoch + 1}")
                break

            epoch_time = time.time() - start_time
            print(f"{phase_name} epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")

        return model_save_path

    def run(self):
        # Regular training mode
        model_path = os.path.join(self.run_dir, "best_model.pth")
        model_path = self.run_training_phase(
            self.train_loader, self.num_epochs, "train",
            model_path, self.config.get('early_stopping_patience')
        )

        # Final testing uses regular training model
        final_model_path = model_path

        # Final testing
        if self.test_loader:
            # Load best model for testing
            self.model.load_state_dict(torch.load(final_model_path))
            test_loss, test_acc = self.evaluate(self.test_loader, "Testing")
            self.history['test_loss'] = test_loss
            if self.config.get('is_classifier', True):
                self.history['test_acc'] = test_acc
            print(f"Test results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        # Save final results
        results = {
            'config': self.config,
            'history': self.history,
            'model_path': final_model_path
        }

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        return results