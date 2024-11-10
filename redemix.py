import librosa
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class EnhancedSpectrogramProcessor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
    def normalize_channel(self, data, scale_factor=1.0):
        """Normalize data to 0-255 range with optional scaling."""
        data_min, data_max = data.min(), data.max()
        normalized = ((data - data_min) / (data_max - data_min) * 255 * scale_factor).clip(0, 255).astype(np.uint8)
        return normalized, (data_min, data_max)
    
    def denormalize_channel(self, normalized_data, params, scale_factor=1.0):
        """Denormalize data back to original range."""
        data_min, data_max = params
        return (normalized_data.astype(float) / (255 * scale_factor)) * (data_max - data_min) + data_min

    def create_complex_spectrogram(self, audio, sr, n_fft=2048, hop_length=512):
        """Create spectrogram with complex component encoding."""
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # Extract components
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        real = np.real(stft)
        imag = np.imag(stft)
        
        # Convert magnitude to dB scale
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Normalize each component
        real_norm, real_params = self.normalize_channel(real, scale_factor=0.8)
        imag_norm, imag_params = self.normalize_channel(imag, scale_factor=0.8)
        mag_norm, mag_params = self.normalize_channel(magnitude_db, scale_factor=1.0)
        phase_norm, phase_params = self.normalize_channel(phase, scale_factor=1.0)
        
        # Store parameters for reconstruction
        self.normalization_params = {
            'real': real_params,
            'imag': imag_params,
            'magnitude': mag_params,
            'phase': phase_params
        }
        
        # Ensure all components are the same size by converting to PIL Images first
        r_img = Image.fromarray(real_norm)
        g_img = Image.fromarray(imag_norm)
        b_img = Image.fromarray(mag_norm)
        a_img = Image.fromarray(phase_norm)
        
        # Store original size for reconstruction
        self.original_size = real_norm.shape
        
        # Create RGBA image by merging channels
        rgba_image = Image.merge('RGBA', (r_img, g_img, b_img, a_img))
        
        return rgba_image

    def process_image_with_diffusion(self, image, strength=0.3, guidance_scale=7.5):
        """Process image while preserving complex information."""
        try:
            # Extract channels
            r, g, b, a = image.split()
            
            # Verify all channels are the same size
            channel_sizes = [channel.size for channel in (r, g, b, a)]
            if len(set(channel_sizes)) != 1:
                raise ValueError(f"Channel sizes don't match: {channel_sizes}")
            
            # Create RGB image for processing
            rgb_image = Image.merge('RGB', (r, g, b))
            
            # Process with diffusion
            processed_rgb = self.img2img(
                image=rgb_image,
                prompt="high quality, detailed audio spectrogram",
                negative_prompt="blurry, distorted, artifacts",
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=30
            ).images[0]
            
            # Extract processed RGB channels
            pr, pg, pb = processed_rgb.split()
            
            # Ensure processed channels match original size
            processed_channels = [pr, pg, pb, a]
            target_size = a.size
            processed_channels = [
                channel.resize(target_size, Image.LANCZOS) 
                for channel in processed_channels
            ]
            
            # Recombine with original phase
            return Image.merge('RGBA', processed_channels)
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise

    def spectrogram_to_audio(self, rgba_image, sr, n_fft=2048, hop_length=512):
        """Convert processed RGBA spectrogram back to audio."""
        # Split into channels
        r, g, b, a = rgba_image.split()
        
        # Convert back to numpy arrays
        rgba_array = np.stack([
            np.array(channel).reshape(self.original_size)
            for channel in (r, g, b, a)
        ], axis=-1)
        
        # Extract components
        real = self.denormalize_channel(rgba_array[..., 0], self.normalization_params['real'], scale_factor=0.8)
        imag = self.denormalize_channel(rgba_array[..., 1], self.normalization_params['imag'], scale_factor=0.8)
        magnitude_db = self.denormalize_channel(rgba_array[..., 2], self.normalization_params['magnitude'])
        phase = self.denormalize_channel(rgba_array[..., 3], self.normalization_params['phase'])
        
        # Convert magnitude back from dB
        magnitude = librosa.db_to_amplitude(magnitude_db)
        
        # Reconstruct complex spectrogram
        stft_complex1 = real + 1j * imag
        stft_complex2 = magnitude * np.exp(1j * phase)
        stft_reconstructed = 0.5 * (stft_complex1 + stft_complex2)
        
        # Inverse STFT
        audio_reconstructed = librosa.istft(stft_reconstructed, 
                                          hop_length=hop_length, 
                                          n_fft=n_fft)
        
        return audio_reconstructed

    def visualize_components(self, rgba_image):
        """Visualize all components of the spectrogram."""
        r, g, b, a = rgba_image.split()
        components = [r, g, b, a]
        titles = ['Real Component', 'Imaginary Component', 
                 'Magnitude (dB)', 'Phase']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, (component, title) in enumerate(zip(components, titles)):
            im = axes[i].imshow(np.array(component), cmap='viridis')
            axes[i].set_title(title)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        return fig

def main():
    processor = EnhancedSpectrogramProcessor()
    
    # Example usage
    input_file = "input.wav"
    output_file = "output_enhanced.wav"
    
    # Load audio
    audio, sr = librosa.load(input_file, sr=None)
    
    # Create complex spectrogram
    spec_image = processor.create_complex_spectrogram(audio, sr)
    
    # Process with conservative settings
    processed_image = processor.process_image_with_diffusion(spec_image, strength=0.15)
    
    # Convert back to audio
    processed_audio = processor.spectrogram_to_audio(processed_image, sr)
    
    # Save result
    librosa.output.write_wav(output_file, processed_audio, sr)
    
    # Visualize components
    processor.visualize_components(spec_image)
    plt.show()

if __name__ == "__main__":
    main()
