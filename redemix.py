import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import warnings
warnings.filterwarnings('ignore')

class AudioSpectrogramProcessor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
    def load_audio(self, file_path, sr=44100):
        """Load audio file and return signal and sample rate."""
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    
    def create_spectrogram_with_phase(self, audio, sr, n_fft=2048, hop_length=512):
        """Create spectrogram with phase information encoded in alpha channel."""
        # Compute complex spectrogram
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Convert magnitude to dB scale
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Normalize magnitude to 0-255 range
        magnitude_normalized = ((magnitude_db - magnitude_db.min()) * 
                              (255 / (magnitude_db.max() - magnitude_db.min()))).astype(np.uint8)
        
        # Normalize phase to 0-255 range (from -π to π)
        phase_normalized = ((phase + np.pi) * (255 / (2 * np.pi))).astype(np.uint8)
        
        # Create RGBA image
        height, width = magnitude_normalized.shape
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[..., 0] = magnitude_normalized  # R channel - magnitude
        rgba_image[..., 1] = magnitude_normalized  # G channel - magnitude
        rgba_image[..., 2] = magnitude_normalized  # B channel - magnitude
        rgba_image[..., 3] = phase_normalized      # A channel - phase
        
        return Image.fromarray(rgba_image, 'RGBA')
    
    def process_image_with_diffusion(self, image, strength=0.3, guidance_scale=7.5):
        """Process image using Stable Diffusion img2img while preserving alpha channel."""
        # Extract alpha channel (phase information)
        alpha = np.array(image.split()[-1])
        
        # Convert to RGB for processing
        rgb_image = image.convert('RGB')
        
        # Process with minimal prompting
        processed_rgb = self.img2img(
            image=rgb_image,
            prompt="high quality, detailed",
            negative_prompt="blurry, distorted",
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=30
        ).images[0]
        
        # Recombine with original alpha channel
        processed_rgba = processed_rgb.convert('RGBA')
        processed_rgba.putalpha(Image.fromarray(alpha))
        
        return processed_rgba
    
    def spectrogram_to_audio(self, rgba_image, sr, n_fft=2048, hop_length=512):
        """Convert RGBA spectrogram back to audio, using alpha channel for phase."""
        # Convert PIL Image to numpy array
        rgba_array = np.array(rgba_image)
        
        # Extract magnitude from RGB (using R channel)
        magnitude_normalized = rgba_array[..., 0]
        
        # Extract phase from alpha channel
        phase_normalized = rgba_array[..., 3]
        
        # Denormalize magnitude
        magnitude_db = (magnitude_normalized / 255.0 * 
                       (self.magnitude_max - self.magnitude_min) + self.magnitude_min)
        magnitude = librosa.db_to_amplitude(magnitude_db)
        
        # Denormalize phase (-π to π)
        phase = (phase_normalized / 255.0 * (2 * np.pi)) - np.pi
        
        # Reconstruct complex spectrogram
        stft_reconstructed = magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        audio_reconstructed = librosa.istft(stft_reconstructed, 
                                          hop_length=hop_length, 
                                          n_fft=n_fft)
        
        return audio_reconstructed
    
    def process_audio_file(self, input_path, output_path, strength=0.3):
        """Complete pipeline to process audio file with phase preservation."""
        # Load audio
        audio, sr = self.load_audio(input_path)
        
        # Create spectrogram with phase
        spec_image = self.create_spectrogram_with_phase(audio, sr)
        
        # Store original magnitude range for reconstruction
        self.magnitude_min = librosa.amplitude_to_db(np.abs(librosa.stft(audio))).min()
        self.magnitude_max = librosa.amplitude_to_db(np.abs(librosa.stft(audio))).max()
        
        # Process with diffusion
        processed_image = self.process_image_with_diffusion(spec_image, strength=strength)
        
        # Convert back to audio
        processed_audio = self.spectrogram_to_audio(processed_image, sr)
        
        # Save processed audio
        librosa.output.write_wav(output_path, processed_audio, sr)
        
        return processed_audio, sr

    def visualize_spectrograms(self, original_image, processed_image):
        """Visualize original and processed spectrograms side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Display original
        ax1.imshow(np.array(original_image)[:,:,:3])  # RGB channels only
        ax1.set_title('Original Spectrogram')
        ax1.axis('off')
        
        # Display processed
        ax2.imshow(np.array(processed_image)[:,:,:3])  # RGB channels only
        ax2.set_title('Processed Spectrogram')
        ax2.axis('off')
        
        plt.tight_layout()
        return fig

def main():
    processor = AudioSpectrogramProcessor()
    
    input_file = "input.wav"
    output_file = "output_processed.wav"
    
    # Process with very conservative settings for initial testing
    processed_audio, sr = processor.process_audio_file(
        input_file,
        output_file,
        strength=0.15  # Very subtle processing
    )
    
    print(f"Processed audio saved to {output_file}")

if __name__ == "__main__":
    main()
