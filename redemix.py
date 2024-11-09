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
        # Initialize the img2img pipeline
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
    def load_audio(self, file_path, sr=44100):
        """Load audio file and return signal and sample rate."""
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    
    def create_spectrogram(self, audio, sr, n_fft=2048, hop_length=512):
        """Create spectrogram from audio signal."""
        # Compute complex spectrogram
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        # Save phase information
        self.phase = np.angle(stft)
        # Get magnitude
        magnitude = np.abs(stft)
        # Convert to dB scale
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        return magnitude_db
    
    def spectrogram_to_image(self, spectrogram):
        """Convert spectrogram to PIL Image."""
        # Normalize to 0-255 range
        spec_normalized = ((spectrogram - spectrogram.min()) * (255 / (spectrogram.max() - spectrogram.min()))).astype(np.uint8)
        # Create RGB image (3 channels)
        spec_rgb = np.stack([spec_normalized] * 3, axis=-1)
        return Image.fromarray(spec_rgb)
    
    def image_to_spectrogram(self, image):
        """Convert PIL Image back to spectrogram."""
        # Convert to grayscale and numpy array
        spec_array = np.array(image.convert('L'))
        # Denormalize
        spec_min, spec_max = spectrogram.min(), spectrogram.max()
        spec_denorm = (spec_array / 255.0) * (spec_max - spec_min) + spec_min
        return spec_denorm
    
    def process_image_with_diffusion(self, image, strength=0.3, guidance_scale=7.5):
        """Process image using Stable Diffusion img2img."""
        # Ensure image is RGB
        image = image.convert('RGB')
        # Process with minimal prompting to maintain structure
        result = self.img2img(
            image=image,
            prompt="high quality, detailed",
            negative_prompt="blurry, distorted",
            strength=strength,  # Lower strength = less change
            guidance_scale=guidance_scale,  # Lower guidance = less creative freedom
            num_inference_steps=30
        ).images[0]
        return result
    
    def spectrogram_to_audio(self, spectrogram, sr, n_fft=2048, hop_length=512):
        """Convert processed spectrogram back to audio."""
        # Convert back to linear scale
        magnitude = librosa.db_to_amplitude(spectrogram)
        # Combine magnitude and phase
        stft_processed = magnitude * np.exp(1j * self.phase)
        # Inverse STFT
        audio_processed = librosa.istft(stft_processed, hop_length=hop_length, n_fft=n_fft)
        return audio_processed
    
    def process_audio_file(self, input_path, output_path, strength=0.3):
        """Complete pipeline to process audio file."""
        # Load audio
        audio, sr = self.load_audio(input_path)
        
        # Create spectrogram
        spectrogram = self.create_spectrogram(audio, sr)
        
        # Convert to image
        spec_image = self.spectrogram_to_image(spectrogram)
        
        # Process with diffusion
        processed_image = self.process_image_with_diffusion(spec_image, strength=strength)
        
        # Convert back to spectrogram
        processed_spec = self.image_to_spectrogram(processed_image)
        
        # Convert back to audio
        processed_audio = self.spectrogram_to_audio(processed_spec, sr)
        
        # Save processed audio
        librosa.output.write_wav(output_path, processed_audio, sr)
        
        return processed_audio, sr

# Example usage
def main():
    processor = AudioSpectrogramProcessor()
    
    # Process an audio file
    input_file = "input.wav"
    output_file = "output_processed.wav"
    
    processed_audio, sr = processor.process_audio_file(
        input_file,
        output_file,
        strength=0.3  # Adjust this value to control how much the diffusion model affects the audio
    )
    
    print(f"Processed audio saved to {output_file}")

if __name__ == "__main__":
    main()
