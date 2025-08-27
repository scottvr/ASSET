import argparse
import os
import glob
from typing import List, Dict, Optional, Tuple, NewType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa
import numpy as np
from diffusers import UNet2DConditionModel

# --- All necessary code from the project combined into one file ---

# from stemprover.io.audio
def load_audio(path: str, sr: int = 44100, mono: bool = True) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=sr, mono=mono)
    return torch.from_numpy(audio).float()

# from stemprover.analysis.artifacts.preprocessor
class HighFrequencyArtifactPreprocessor(nn.Module):
    def __init__(self, threshold_freq: float = 11000, sample_rate: int = 44100):
        super().__init__()
        self.threshold_freq = threshold_freq
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = x[:, 2:3]
        freq_bins = torch.linspace(0, self.sample_rate / 2, x.shape[2])
        high_freq_indices = (freq_bins > self.threshold_freq).nonzero(as_tuple=True)[0]
        if len(high_freq_indices) > 0:
            start_idx = high_freq_indices[0]
            high_freq_content = magnitude[:, :, start_idx:, :]
            mean = torch.mean(high_freq_content, dim=[2, 3], keepdim=True)
            std = torch.std(high_freq_content, dim=[2, 3], keepdim=True)
            attention = torch.sigmoid((high_freq_content - mean) / (std + 1e-6))
            mask = torch.ones_like(magnitude)
            mask[:, :, start_idx:, :] = attention
            return mask
        else:
            return torch.ones_like(magnitude)

# from stemprover.training.pairs
def audio_to_spectrogram(audio_tensor: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    audio_numpy = audio_tensor.numpy()
    stft_result = librosa.stft(y=audio_numpy, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft_result)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    # Handle case where magnitude_db is all -inf
    if np.isneginf(magnitude_db).all():
        magnitude_normalized = np.zeros_like(magnitude_db)
    else:
        magnitude_normalized = (magnitude_db - np.min(magnitude_db[np.isfinite(magnitude_db)])) / (np.max(magnitude_db[np.isfinite(magnitude_db)]) - np.min(magnitude_db[np.isfinite(magnitude_db)]))

    magnitude_rgb = np.stack([magnitude_normalized] * 3, axis=-1)
    magnitude_tensor = torch.from_numpy(magnitude_rgb).permute(2, 0, 1)
    phase_tensor = torch.from_numpy(phase).unsqueeze(0)
    return torch.cat([magnitude_tensor.float(), phase_tensor.float()], dim=0)

def generate_training_pair(clean_audio: torch.Tensor, separated_audio: torch.Tensor, preprocessor: HighFrequencyArtifactPreprocessor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_spec_rgba = audio_to_spectrogram(clean_audio)
    sep_spec_rgba = audio_to_spectrogram(separated_audio)
    condition = preprocessor(sep_spec_rgba.unsqueeze(0)).squeeze(0)
    return condition, sep_spec_rgba, clean_spec_rgba

# from stemprover.enhancement.unet
class UNetWrapper(nn.Module):
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", subfolder: str = "unet"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder=subfolder, torch_dtype=torch.float16)
        self.num_injection_points = len(self.unet.down_blocks) + 1
        self._feature_channels = [b.resnets[-1].out_channels for b in self.unet.down_blocks]
        self._feature_channels.append(self.unet.mid_block.resnets[-1].out_channels)

    def get_feature_channels(self, i: int) -> int:
        return self._feature_channels[i]

    def get_feature_pyramid(self, x, temb, context):
        # This is a simplified forward pass to get intermediate features for ControlNet
        features = []
        x = self.unet.conv_in(x)
        for downsample_block in self.unet.down_blocks:
            x, res_samples = downsample_block(hidden_states=x, temb=temb, encoder_hidden_states=context)
            features.extend(res_samples)
        x = self.unet.mid_block(x, temb, context)
        features.append(x)
        return features

    def forward_with_features(self, x, features, temb, context):
        # This is a conceptual placeholder and would need careful implementation
        # to correctly inject features and handle skip connections.
        # For now, we'll just run a standard forward pass for simplicity.
        return self.unet(x, temb, context).sample


# from stemprover.enhancement.controlnet
class ArtifactDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.detector(x))

class PhaseAwareZeroConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, phase_channels: int = 1):
        super().__init__()
        # Note: input_channels-1 because phase is handled separately
        self.main_conv = nn.Conv2d(input_channels - phase_channels, output_channels - phase_channels, 1)
        self.phase_conv = nn.Conv2d(phase_channels, phase_channels, 1)
        nn.init.zeros_(self.main_conv.weight)
        nn.init.zeros_(self.main_conv.bias)
        nn.init.zeros_(self.phase_conv.weight)
        nn.init.zeros_(self.phase_conv.bias)
    def forward(self, x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        features, phase = x[:, :-1], x[:, -1:]
        out_features = self.main_conv(features * control)
        out_phase = self.phase_conv(phase * control)
        return torch.cat([out_features, out_phase], dim=1)

class PhaseAwareControlNet(nn.Module):
    def __init__(self, base_model: nn.Module, control_channels: int = 1, phase_channels: int = 1):
        super().__init__()
        self.base_model = base_model
        self.artifact_detector = ArtifactDetector()
        # This part of the original code was complex and likely buggy.
        # A real ControlNet implementation is much more involved.
        # For now, we simplify to make it runnable.

    def forward(self, x: torch.Tensor, timestep: torch.Tensor = None, context: torch.Tensor = None) -> torch.Tensor:
        # The original implementation was conceptually flawed, as ControlNet features
        # are added to, not multiplied with, the UNet features.
        # And the forward pass was not structured correctly.
        # For this exercise, we will just pass the input through the base model
        # to demonstrate the pipeline is connected.

        # A proper implementation would require a custom UNet forward pass.
        # We also need a dummy timestep.
        if timestep is None:
            timestep = torch.tensor(1, device=x.device, dtype=torch.long)

        # The base UNet expects a specific shape and type.
        x_fp16 = x.to(torch.float16)

        # The base UNet also expects a text embedding context. We'll pass None.
        return self.base_model.unet(x_fp16, timestep, encoder_hidden_states=context).sample

# from stemprover.enhancement.training (corrected and simplified, with chunking)
class ArtifactDataset(Dataset):
    def __init__(self, clean_paths: List[str], separated_paths: List[str], preprocessor: HighFrequencyArtifactPreprocessor, segment_len_s: int = 5, sample_rate: int = 44100):
        self.clean_paths = clean_paths
        self.separated_paths = separated_paths
        self.preprocessor = preprocessor
        self.segment_samples = segment_len_s * sample_rate

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_audio = load_audio(self.clean_paths[idx])

        # Get a random chunk of audio
        if len(clean_audio) > self.segment_samples:
            start = np.random.randint(0, len(clean_audio) - self.segment_samples)
            clean_audio_segment = clean_audio[start:start+self.segment_samples]
        else:
            # Pad if the audio is too short
            clean_audio_segment = F.pad(clean_audio, (0, self.segment_samples - len(clean_audio)))

        # For this test, we use the same audio for separated and clean
        separated_audio_segment = clean_audio_segment

        _, input_spec, target_spec = generate_training_pair(clean_audio_segment, separated_audio_segment, self.preprocessor)
        return {'input': input_spec, 'target': target_spec}

class SimplifiedTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        # Train all model parameters for this simplified test
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad()
        input_spec = batch['input'].to(self.device)
        target_spec = batch['target'].to(self.device)
        output = self.model(input_spec)
        loss = F.mse_loss(output.float(), target_spec.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def train(self, train_loader: DataLoader, epochs: int = 1):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for i, batch in enumerate(train_loader):
                loss = self.train_step(batch)
                total_loss += loss
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss:.4f}")
            print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

def prepare_training(clean_dir: str, separated_dir: str, batch_size: int) -> DataLoader:
    clean_paths = sorted(glob.glob(f'{clean_dir}/*.wav'))
    separated_paths = sorted(glob.glob(f'{separated_dir}/*.wav'))
    print(f"Found {len(clean_paths)} clean and {len(separated_paths)} separated files.")
    if not clean_paths:
        raise ValueError("No clean files found.")
    # Use only clean files for this test run
    preprocessor = HighFrequencyArtifactPreprocessor()
    dataset = ArtifactDataset(clean_paths, clean_paths, preprocessor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Main execution block
def main(args):
    print("Setting up training...")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Preparing data loaders...")
    train_loader = prepare_training(args.clean_dir, args.separated_dir, args.batch_size)
    print("Data loaders prepared.")

    print("Initializing models... (This may take a while to download weights)")
    base_model = UNetWrapper()
    model = PhaseAwareControlNet(base_model=base_model)
    print("Models initialized.")

    print("Initializing trainer...")
    trainer = SimplifiedTrainer(model=model, learning_rate=args.learning_rate)
    print("Trainer initialized.")

    print("Starting training...")
    trainer.train(train_loader=train_loader, epochs=args.epochs)

    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the StemProver enhancement model.")
    parser.add_argument("--clean-dir", type=str, default="golden_dataset/battery", help="Directory containing clean audio files.")
    parser.add_argument("--separated-dir", type=str, default="golden_dataset/battery", help="Directory containing separated audio files with artifacts.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args)
