import argparse
import os
import csv
import time
from datetime import datetime
import torch
import torch.nn as nn
from train import Runner


def run_experiment(model_type, subject, data_dir, run_dir, batch_size, val_split, seed):
    """Run single experiment"""
    # Create save dir
    model_save_dir = os.path.join(run_dir, model_type, f"subject_{subject}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Determine feature types
    feature_type = "time_domain" if "time_domain" in model_type else "stft"

    # Create datasets
    # train_dataset, test_dataset = create_subject_datasets(subject, data_dir, model_type)
    debug = True
    # seed = random.randint(0, 114514)
    # print(f"seed: {seed}")
    # Experimental Config
    config = {
        'run_dir': model_save_dir,
        'data_dir': data_dir,
        'subject': subject,
        'model_type': model_type,
        'optimizer': {
            'class': torch.optim.AdamW,
            'params': {'lr': 0.001, 'weight_decay': 1e-5},
        },
        'criterion': nn.CrossEntropyLoss,
        'batch_size': batch_size,
        'val_split': val_split,
        'early_stopping_patience': 10,
        'clip_grad': None,
        'is_classifier': True,
        'num_workers': 0,
        'seed': seed,
    }

    # Run experiments
    runner = Runner(config)
    results = runner.run()

    # Record results
    test_acc = results['history']['test_acc']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(run_dir, f"exp_grid_results.csv")

    file_exists = os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'model_type', 'subject', 'test_acc'])
        writer.writerow([timestamp, model_type, subject, test_acc])

    print(f"Model {model_type} - Subject {subject} test complete. Accuracy: {test_acc:.4f}")
    return test_acc


def main():
    parser = argparse.ArgumentParser(description='Experiment Grid: Compare Different Models')
    parser.add_argument('--data_dir', default='./data/DB2', help='Dataset directory')
    parser.add_argument('--run_dir', default='./runs/exp_grid', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    args = parser.parse_args([])

    # Model comparisons
    model_types = [
        # 'sinca_s',
        'sinca_xs',
        'sinca_xxs',
        # "stft+cacnn",
    ]

    # Set seed
    # set_seed(42)
    subjects = list(range(1, 41))
    val_splits = [0.2] # val_splits = [0, 0.2, 0.4, 0.6, 0.8]
    # Run models, subjects and validation splits
    for model_type in model_types:
        for seed in [0, 42, 1234, 2026]:
            for val_split in val_splits:
                for subject in subjects:
                    print(f"\n{'=' * 50}")
                    print(f"Running Model: {model_type} - Subject: {subject}")
                    print(f"{'=' * 50}")

                    start_time = time.time()
                    test_acc = run_experiment(
                        model_type=model_type,
                        subject=subject,
                        data_dir=args.data_dir,
                        run_dir=args.run_dir,
                        batch_size=args.batch_size,
                        val_split=val_split,
                        seed=seed
                    )
                    elapsed = time.time() - start_time
                    print(f"Completed in {elapsed:.2f} seconds. Accuracy: {test_acc:.4f}")
    print("\nExperiment Grid Complete!")
    print(f"Results saved to: {os.path.join(args.run_dir, 'exp_grid_results.csv')}")


if __name__ == "__main__":
    main()
