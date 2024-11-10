import librosa
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
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
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None
        ).to(device)
        
        # Initialize Real-ESRGAN
        self.upsampler = self._initialize_real_esrgan()
        
        # Enable memory efficient attention and VAE tiling
        self.img2img.enable_attention_slicing(slice_size=1)
        self.img2img.enable_vae_tiling()
        
        # Clear CUDA cache
        if device == "cuda":
            torch.cuda.empty_cache()

    def _initialize_real_esrgan(self):
        """Initialize Real-ESRGAN model for upscaling."""
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Download model weights if not present
        model_path = load_file_from_url(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model_dir='weights'
        )
        
        # Initialize upsampler
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,  # Process in tiles to save memory
            tile_pad=32,
            pre_pad=0,
            half=True if self.device == "cuda" else False
        )
        
        return upsampler

    # ... (previous normalize_channel and denormalize_channel methods remain the same)

    def process_image_with_diffusion(self, image, strength=0.3, guidance_scale=7.5, chunk_size=512, scale_factor=0.5):
        """Process image in chunks with downscaling and upscaling."""
        try:
            # Extract channels
            r, g, b, a = image.split()
            
            # Get original size
            original_width, original_height = image.size
            
            # Calculate new size for processing
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Downscale RGB channels
            rgb_image = Image.merge('RGB', (r, g, b))
            rgb_small = rgb_image.resize((new_width, new_height), Image.LANCZOS)
            del rgb_image
            
            # Process in vertical chunks
            processed_chunks = []
            
            for y in range(0, new_height, chunk_size):
                # Extract chunk
                chunk_height = min(chunk_size, new_height - y)
                chunk_box = (0, y, new_width, y + chunk_height)
                chunk_rgb = rgb_small.crop(chunk_box)
                
                # Process chunk with diffusion
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
            combined_rgb = Image.new('RGB', (new_width, total_height))
            
            y_offset = 0
            for chunk in processed_chunks:
                combined_rgb.paste(chunk, (0, y_offset))
                y_offset += chunk.height
                del chunk
            
            # Upscale with Real-ESRGAN
            print("Upscaling with Real-ESRGAN...")
            upscaled_rgb, _ = self.upsampler.enhance(
                np.array(combined_rgb),
                outscale=original_width / new_width  # Calculate required scale factor
            )
            del combined_rgb
            
            # Convert back to PIL Image
            upscaled_rgb = Image.fromarray(upscaled_rgb)
            
            # Ensure exact size match
            if upscaled_rgb.size != (original_width, original_height):
                upscaled_rgb = upscaled_rgb.resize((original_width, original_height), Image.LANCZOS)
            
            # Extract processed RGB channels
            pr, pg, pb = upscaled_rgb.split()
            del upscaled_rgb
            
            # Recombine with original phase
            result = Image.merge('RGBA', (pr, pg, pb, a))
            
            # Clear processed channels
            del pr, pg, pb
            
            return result
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise

    # ... (previous methods remain the same)

def main():
    # Set PyTorch memory allocation configuration
    torch.cuda.set_per_process_memory_fraction(0.7)
    
    processor = EnhancedSpectrogramProcessor()
    
    # Example usage
    input_file = "input.wav"
    output_file = "output_enhanced.wav"
    
    # Load audio
    audio, sr = librosa.load(input_file, sr=None)
    
    # Create complex spectrogram
    spec_image = processor.create_complex_spectrogram(audio, sr)
    
    # Process with conservative settings, smaller chunks, and scaling
    processed_image = processor.process_image_with_diffusion(
        spec_image, 
        strength=0.15,
        chunk_size=512,
        scale_factor=0.5  # Process at half resolution
    )
    
    # Convert back to audio
    processed_audio = processor.spectrogram_to_audio(
        processed_image, 
        sr,
        chunk_size=512
    )
    
    # Save result
    librosa.output.write_wav(output_file, processed_audio, sr)
    
    # Visualize components (with size limit)
    processor.visualize_components(spec_image)
    plt.show()

if __name__ == "__main__":
    main()
