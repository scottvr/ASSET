import argparse
import os
import torch

from src.stemprover.enhancement.controlnet import PhaseAwareControlNet
from src.stemprover.enhancement.unet import UNetWrapper
from src.stemprover.enhancement.training import ControlNetTrainer, prepare_training
from src.stemprover.enhancement.base import HighFrequencyArtifactPreprocessor


def main(args):
    """
    Main training function.
    """
    print("Setting up training...")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Prepare DataLoaders
    print("Preparing data loaders...")
    train_loader, val_loader = prepare_training(
        clean_dir=args.clean_dir,
        separated_dir=args.separated_dir,
        batch_size=args.batch_size
    )
    print("Data loaders prepared.")

    # 2. Initialize Model
    print("Initializing models... (This may take a while to download weights)")
    base_model = UNetWrapper()
    model = PhaseAwareControlNet(base_model=base_model)
    print("Models initialized.")

    # 3. Initialize Trainer
    print("Initializing trainer...")
    preprocessor = HighFrequencyArtifactPreprocessor()
    trainer = ControlNetTrainer(
        model=model,
        preprocessor=preprocessor,
        learning_rate=args.learning_rate,
        device=args.device
    )
    print("Trainer initialized.")

    # 4. Start Training
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir
    )

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the StemProver enhancement model.")

    parser.add_argument("--clean-dir", type=str, required=True, help="Directory containing clean audio files.")
    parser.add_argument("--separated-dir", type=str, required=True, help="Directory containing separated audio files with artifacts.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")

    args = parser.parse_args()
    main(args)
