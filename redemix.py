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
        # Enable memory efficient attention
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,  # More memory efficient model loading
            variant="fp16" if device == "cuda" else None
        ).to(device)
        
        # Enable memory efficient attention and VAE tiling
        self.img2img.enable_attention_slicing(slice_size=1)
        self.img2img.enable_vae_tiling()
        
        # Clear CUDA cache
        if device == "cuda":
            torch.cuda.empty_cache()
    
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
        # Process in smaller chunks if audio is too long
        chunk_size = sr * 60  # Process 60 seconds at a time
        if len(audio) > chunk_size:
            spectrograms = []
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                spec = self._process_audio_chunk(chunk, sr, n_fft, hop_length)
                spectrograms.append(spec)
            return self._combine_spectrograms(spectrograms)
        else:
            return self._process_audio_chunk(audio, sr, n_fft, hop_length)

    def _process_audio_chunk(self, audio_chunk, sr, n_fft, hop_length):
        """Process a single chunk of audio data."""
        # Compute STFT
        stft = librosa.stft(audio_chunk, n_fft=n_fft, hop_length=hop_length)
        
        # Extract components one at a time to reduce memory usage
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        real = np.real(stft)
        imag = np.imag(stft)
        
        # Free memory
        del stft
        
        # Convert magnitude to dB scale
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        del magnitude
        
        # Normalize components
        real_norm, real_params = self.normalize_channel(real, scale_factor=0.8)
        del real
        imag_norm, imag_params = self.normalize_channel(imag, scale_factor=0.8)
        del imag
        mag_norm, mag_params = self.normalize_channel(magnitude_db, scale_factor=1.0)
        del magnitude_db
        phase_norm, phase_params = self.normalize_channel(phase, scale_factor=1.0)
        del phase
        
        # Store parameters for reconstruction
        self.normalization_params = {
            'real': real_params,
            'imag': imag_params,
            'magnitude': mag_params,
            'phase': phase_params
        }
        
        # Create image channels
        r_img = Image.fromarray(real_norm)
        del real_norm
        g_img = Image.fromarray(imag_norm)
        del imag_norm
        b_img = Image.fromarray(mag_norm)
        del mag_norm
        a_img = Image.fromarray(phase_norm)
        del phase_norm
        
        # Store original size for reconstruction
        self.original_size = r_img.size[::-1]
        
        # Create RGBA image
        return Image.merge('RGBA', (r_img, g_img, b_img, a_img))

    def _combine_spectrograms(self, spectrograms):
        """Combine multiple spectrogram chunks."""
        # Assuming vertical stacking of spectrograms
        total_height = sum(spec.height for spec in spectrograms)
        combined = Image.new('RGBA', (spectrograms[0].width, total_height))
        
        y_offset = 0
        for spec in spectrograms:
            combined.paste(spec, (0, y_offset))
            y_offset += spec.height
            del spec  # Free memory
        
        return combined

    def process_image_with_diffusion(self, image, strength=0.3, guidance_scale=7.5, chunk_size=512):
        """Process image in chunks to reduce memory usage."""
        try:
            # Extract channels
            r, g, b, a = image.split()
            
            # Process in vertical chunks
            width, height = image.size
            processed_chunks = []
            
            for y in range(0, height, chunk_size):
                # Extract chunk
                chunk_height = min(chunk_size, height - y)
                chunk_box = (0, y, width, y + chunk_height)
                
                chunk_rgb = Image.merge('RGB', (
                    r.crop(chunk_box),
                    g.crop(chunk_box),
                    b.crop(chunk_box)
                ))
                
                # Process chunk
                processed_chunk = self.img2img(
                    image=chunk_rgb,
                    prompt="high quality, detailed audio spectrogram",
                    negative_prompt="blurry, distorted, artifacts",
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=30
                ).images[0]
                
                processed_chunks.append(processed_chunk)
                
                # Clear CUDA cache after each chunk
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Combine processed chunks
            total_height = sum(chunk.height for chunk in processed_chunks)
            combined_rgb = Image.new('RGB', (width, total_height))
            
            y_offset = 0
            for chunk in processed_chunks:
                combined_rgb.paste(chunk, (0, y_offset))
                y_offset += chunk.height
                del chunk  # Free memory
            
            # Extract processed RGB channels
            pr, pg, pb = combined_rgb.split()
            del combined_rgb
            
            # Ensure processed channels match original size
            processed_channels = [pr, pg, pb, a]
            target_size = a.size
            processed_channels = [
                channel.resize(target_size, Image.LANCZOS) 
                for channel in processed_channels
            ]
            
            # Recombine with original phase
            result = Image.merge('RGBA', processed_channels)
            
            # Clear processed channels
            del processed_channels
            
            return result
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise

    def spectrogram_to_audio(self, rgba_image, sr, n_fft=2048, hop_length=512, chunk_size=512):
        """Convert processed RGBA spectrogram back to audio in chunks."""
        # Split into channels
        r, g, b, a = rgba_image.split()
        width, height = rgba_image.size
        
        audio_chunks = []
        
        for y in range(0, height, chunk_size):
            chunk_height = min(chunk_size, height - y)
            chunk_box = (0, y, width, y + chunk_height)
            
            # Process chunk
            chunk_arrays = np.stack([
                np.array(channel.crop(chunk_box))
                for channel in (r, g, b, a)
            ], axis=-1)
            
            # Extract and denormalize components
            real = self.denormalize_channel(chunk_arrays[..., 0], self.normalization_params['real'], scale_factor=0.8)
            imag = self.denormalize_channel(chunk_arrays[..., 1], self.normalization_params['imag'], scale_factor=0.8)
            magnitude_db = self.denormalize_channel(chunk_arrays[..., 2], self.normalization_params['magnitude'])
            phase = self.denormalize_channel(chunk_arrays[..., 3], self.normalization_params['phase'])
            
            del chunk_arrays
            
            # Convert magnitude back from dB
            magnitude = librosa.db_to_amplitude(magnitude_db)
            del magnitude_db
            
            # Reconstruct complex spectrogram
            stft_complex1 = real + 1j * imag
            del real, imag
            stft_complex2 = magnitude * np.exp(1j * phase)
            del magnitude, phase
            
            stft_reconstructed = 0.5 * (stft_complex1 + stft_complex2)
            del stft_complex1, stft_complex2
            
            # Inverse STFT
            audio_chunk = librosa.istft(stft_reconstructed, 
                                      hop_length=hop_length, 
                                      n_fft=n_fft)
            
            audio_chunks.append(audio_chunk)
            
            # Clear memory
            del stft_reconstructed
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Combine audio chunks
        return np.concatenate(audio_chunks)

    def visualize_components(self, rgba_image):
        """Visualize all components of the spectrogram."""
        # Process visualization in smaller chunks if image is too large
        if rgba_image.height > 2048 or rgba_image.width > 2048:
            rgba_image = rgba_image.resize((min(rgba_image.width, 2048), 
                                          min(rgba_image.height, 2048)),
                                         Image.LANCZOS)
        
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
    # Set PyTorch memory allocation configuration
    torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of available GPU memory
    
    processor = EnhancedSpectrogramProcessor()
    
    # Example usage
    input_file = "input.wav"
    output_file = "output_enhanced.wav"
    
    # Load audio
    audio, sr = librosa.load(input_file, sr=None)
    
    # Create complex spectrogram
    spec_image = processor.create_complex_spectrogram(audio, sr)
    
    # Process with conservative settings and smaller chunks
    processed_image = processor.process_image_with_diffusion(
        spec_image, 
        strength=0.15,
        chunk_size=512  # Process in smaller chunks
    )
    
    # Convert back to audio
    processed_audio = processor.spectrogram_to_audio(
        processed_image, 
        sr,
        chunk_size=512  # Process in smaller chunks
    )
    
    # Save result
    librosa.output.write_wav(output_file, processed_audio, sr)
    
    # Visualize components (with size limit)
    processor.visualize_components(spec_image)
    plt.show()

if __name__ == "__main__":
    main()
