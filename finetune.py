import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import argparse
import csv
from datetime import datetime
from model import STFTImitNet, ChannelAttentionCNN
from train import train_model
from data_utils import load_and_seg_data, extract_stft_features, sEMGDataset, CNNFeatureDataset, set_seed
import gc
from torch.utils.tensorboard import SummaryWriter


def finetune_on_subject(subject, data_dir, run_dir, pretrain_dir, batch_size, num_epochs):
    set_seed(42)

    dataset_path = os.path.join(data_dir, f"DB2_s{subject}", f"S{subject}_E1_A1.mat")
    model_save_dir = os.path.join(run_dir, f"DB2_s{subject}")
    os.makedirs(model_save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(model_save_dir, f"tensorboard_{timestamp}")
    writer = SummaryWriter(log_dir=tb_log_dir)

    csv_path = os.path.join(model_save_dir, "finetuning_results.csv")

    # 直接设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stft_imit_net = STFTImitNet(input_length=600, window_size=48).to(device)
    classifier = ChannelAttentionCNN(num_classes=17, input_channels=12).to(device)

    stft_imit_net_path = os.path.join(pretrain_dir, "pretrained_sin_48_1-25.pth")
    stft_imit_net.load_state_dict(torch.load(stft_imit_net_path))

    print(f"\nProcessing subject: {subject}")
    print(f"Dataset path: {dataset_path}")

    X_finetune_raw, y_finetune_raw = load_and_seg_data(dataset_path, [6], window_size=600, step_size=20)
    X_test_raw, y_test_raw = load_and_seg_data(dataset_path, [2, 5], window_size=600, step_size=20)

    mean_train = np.mean(X_finetune_raw, axis=(0, 1), keepdims=True)
    std_train = np.std(X_finetune_raw, axis=(0, 1), keepdims=True) + 1e-8
    X_finetune_scaled = (X_finetune_raw - mean_train) / std_train
    X_test_scaled = (X_test_raw - mean_train) / std_train

    X_finetune_stft, _, _ = extract_stft_features(X_finetune_scaled)
    mean_stft = np.mean(X_finetune_stft, axis=(0, 1, 2), keepdims=True)
    std_stft = np.std(X_finetune_stft, axis=(0, 1, 2), keepdims=True) + 1e-8
    X_finetune_stft_scaled = (X_finetune_stft - mean_stft) / (std_stft + 1e-8)

    finetune_dataset = sEMGDataset(X_finetune_scaled, y_finetune_raw)
    classifier_finetune_dataset = CNNFeatureDataset(X_finetune_stft_scaled, y_finetune_raw)
    test_dataset = sEMGDataset(X_test_scaled, y_test_raw)

    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)
    classifier_finetune_loader = DataLoader(classifier_finetune_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'stage', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])

        print("\n--- Pretraining CNN Classifier on STFT Features ---")
        classifier_optimizer = optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-5)
        classifier_criterion = nn.CrossEntropyLoss()
        classifier_save_path = os.path.join(model_save_dir, "pretrained_classifier.pth")
        train_model(
            None, classifier, classifier_finetune_loader, None,
            classifier_optimizer, classifier_criterion, 30,
            classifier_save_path, writer, csv_writer, 'pretraining', device
            # feature_model defaults to None, indicating single model training
        )
        classifier.load_state_dict(torch.load(classifier_save_path))

        print("\n--- Fine-tuning Combined Model ---")
        combined_optimizer = optim.AdamW(
            list(stft_imit_net.parameters()) + list(classifier.parameters()),
            lr=0.0001, weight_decay=1e-6
        )
        combined_criterion = nn.CrossEntropyLoss()
        combined_save_path = os.path.join(model_save_dir, "best_finetuned_model.pth")

        best_finetune_acc = train_model(
            stft_imit_net, classifier, finetune_loader, None,
            combined_optimizer, combined_criterion, 10,
            combined_save_path, writer, csv_writer, 'finetuning', device
        )

        print("\n--- Final Evaluation on Test Set ---")
        checkpoint = torch.load(combined_save_path)
        stft_imit_net.load_state_dict(checkpoint['stft_imit_net'])
        classifier.load_state_dict(checkpoint['classifier'])

        stft_imit_net.eval()
        classifier.eval()
        test_accs = []

        with torch.no_grad():
            for raw_data, labels in test_loader:
                raw_data, labels = raw_data.to(device), labels.to(device)
                features = stft_imit_net(raw_data)
                outputs = classifier(features)
                _, preds = torch.max(outputs, dim=1)
                acc = torch.sum(preds == labels).item() / len(preds)
                test_accs.append(acc)

        best_test_acc = np.mean(test_accs)

    summary_csv_path = os.path.join(run_dir, "finetune_summary_results.csv")
    summary_exists = os.path.exists(summary_csv_path)

    with open(summary_csv_path, 'a', newline='') as summary_file:
        summary_writer = csv.writer(summary_file)

        if not summary_exists:
            summary_writer.writerow(['subject', 'best_finetune_acc', 'best_test_acc', 'timestamp'])

        summary_writer.writerow([subject, best_finetune_acc, best_test_acc, timestamp])

    print(f"\nSubject {subject} fine-tuning complete!")
    print(f"Best Fine-tuned Model Test Acc: {best_test_acc:.4f}")

    writer.close()

    del stft_imit_net, classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description='Fine-tune STFT-CNN models on individual subjects')
    parser.add_argument('--data_dir', default='./data/DB2',
                        help='Base directory for datasets')
    parser.add_argument('--run_dir', default='./runs/finetune/',
                        help='Base directory to save fine-tuned models')
    parser.add_argument('--pretrain_dir', default='./save/pretrain/',
                        help='Directory containing pretrained models')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Maximum epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=300,
                        help='Batch size for training')
    args = parser.parse_args()

    subjects = list(range(1, 26))
    print(f"Fine-tuning on {len(subjects)} subjects: {subjects}")

    for i, subject in enumerate(subjects):
        print(f"\n{'=' * 50}")
        print(f"Fine-tuning subject {i + 1}/{len(subjects)}: {subject}")
        print(f"{'=' * 50}")

        finetune_on_subject(
            subject=subject,
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            pretrain_dir=args.pretrain_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )

    print(f"Summary results saved to: {os.path.join(args.run_dir, 'finetune_summary_results.csv')}")


if __name__ == "__main__":
    main()