import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
from spleeter.separator import Separator  # We'll use Spleeter for initial testing

class MOGGStereoProcessor:
    def __init__(self, output_dir: str = "test_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize separator
        self.separator = Separator('spleeter:2stems')  # Simple 2-stem separation to start
        
    def load_stereo_pair(self, 
                        track1_path: str, 
                        track2_path: str,
                        sr: int = 44100) -> np.ndarray:
        """Load two mono tracks as stereo pair"""
        # Load mono tracks
        left, sr1 = librosa.load(track1_path, sr=sr, mono=True)
        right, sr2 = librosa.load(track2_path, sr=sr, mono=True)
        
        # Ensure same length
        min_length = min(len(left), len(right))
        left = left[:min_length]
        right = right[:min_length]
        
        # Stack to stereo
        return np.vstack([left, right])
    
    def create_test_pair(self,
                        vocal_left_path: str,
                        vocal_right_path: str,
                        guitar_left_path: str,
                        guitar_right_path: str,
                        duration: float = 30.0) -> dict:
        """Create clean/artifacted pair from MOGG tracks"""
        
        print("Loading vocal tracks...")
        vocal_stereo = self.load_stereo_pair(vocal_left_path, vocal_right_path)
        
        print("Loading guitar tracks...")
        guitar_stereo = self.load_stereo_pair(guitar_left_path, guitar_right_path)
        
        # Trim to specified duration if needed
        if duration:
            samples = int(duration * 44100)
            vocal_stereo = vocal_stereo[:, :samples]
            guitar_stereo = guitar_stereo[:, :samples]
        
        # Create mix
        print("Creating mix...")
        mix_stereo = vocal_stereo + guitar_stereo
        
        # Convert to mono for separation (if needed by separator)
        mix_mono = librosa.to_mono(mix_stereo)
        
        # Separate
        print("Running separation...")
        separated = self.separator.separate(mix_mono)
        artifacted_vocal = separated['vocals']
        
        # Save all versions
        print("Saving files...")
        test_files = self._save_test_files(
            clean_vocal=vocal_stereo,
            guitar=guitar_stereo,
            mix=mix_stereo,
            separated_vocal=artifacted_vocal
        )
        
        return {
            'clean_vocal': vocal_stereo,
            'guitar': guitar_stereo,
            'mix': mix_stereo,
            'artifacted_vocal': artifacted_vocal,
            'file_paths': test_files
        }
    
    def _save_test_files(self,
                        clean_vocal: np.ndarray,
                        guitar: np.ndarray,
                        mix: np.ndarray,
                        separated_vocal: np.ndarray) -> dict:
        """Save all audio files for the test case"""
        
        files = {}
        
        # Create timestamped directory
        from datetime import datetime
        test_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir.mkdir(exist_ok=True)
        
        # Save each file
        for name, audio in [
            ('clean_vocal', clean_vocal),
            ('guitar', guitar),
            ('mix', mix),
            ('separated_vocal', separated_vocal)
        ]:
            path = test_dir / f"{name}.wav"
            sf.write(path, audio.T, 44100)  # .T to convert to soundfile's expected format
            files[name] = path
            
        return files

def main():
    # Paths for Battery MOGG tracks
    vocal_left = "track-09.wav"    # Adjust paths as needed
    vocal_right = "track-10.wav"
    guitar_left = "track-07.wav"
    guitar_right = "track-08.wav"
    
    processor = MOGGStereoProcessor()
    
    print("Creating test pair...")
    result = processor.create_test_pair(
        vocal_left_path=vocal_left,
        vocal_right_path=vocal_right,
        guitar_left_path=guitar_left,
        guitar_right_path=guitar_right,
        duration=30.0  # Start with 30 seconds
    )
    
    print("\nFiles created:")
    for name, path in result['file_paths'].items():
        print(f"{name}: {path}")

if __name__ == "__main__":
    main()
