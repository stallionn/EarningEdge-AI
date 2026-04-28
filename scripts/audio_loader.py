"""
Audio Loading and Standardization Module

Handles loading earnings call audio in multiple formats and standardizing
to 16kHz mono WAV format for consistent processing.

Supported formats: WAV, MP3, M4A, FLAC, OGG
Output format: 16kHz, mono, float32, normalized to [-1, 1]
"""

import numpy as np
import librosa
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFormatError(Exception):
    """Raised when audio format is unsupported or file is corrupted."""
    pass


class AudioDurationError(Exception):
    """Raised when audio duration is outside valid range."""
    pass


class AudioLoadingError(Exception):
    """Raised when audio loading fails for any reason."""
    pass


# Constants
SUPPORTED_AUDIO_FORMATS = {
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg', 
    '.m4a': 'audio/mp4',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg'
}

TARGET_SAMPLE_RATE = 16000  # Hz (standard for speech processing)
TARGET_CHANNELS = 1         # Mono
TARGET_DTYPE = np.float32   # Normalized [-1, 1]

# Duration constraints (earnings calls are typically 30-60 minutes)
MIN_AUDIO_DURATION_SEC = 300        # 5 minutes minimum
MAX_AUDIO_DURATION_SEC = 14400      # 4 hours maximum
TYPICAL_CALL_DURATION_MIN = 2400    # 40 minutes typical


def validate_audio_format(audio_path: str) -> None:
    """
    Validate that audio file has supported format.
    
    Args:
        audio_path: Path to audio file
        
    Raises:
        AudioFormatError: If format not supported
    """
    file_ext = Path(audio_path).suffix.lower()
    
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise AudioFormatError(
            f"Unsupported audio format: {file_ext}\n"
            f"Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS.keys())}"
        )
    
    if not Path(audio_path).exists():
        raise AudioFormatError(f"Audio file not found: {audio_path}")
    
    # Check file size (rough sanity check)
    file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
    if file_size_mb < 1:
        logger.warning(f"Very small audio file ({file_size_mb:.2f} MB) - may be corrupted")
    if file_size_mb > 2000:
        logger.warning(f"Very large audio file ({file_size_mb:.2f} MB) - may take time to load")


def load_and_standardize_audio(
    audio_path: str,
    sr: Optional[int] = TARGET_SAMPLE_RATE,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio from any supported format and convert to standard format.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (default: 16000 Hz)
        mono: Convert to mono if True (default: True)
        
    Returns:
        Tuple of (audio_waveform, sample_rate)
        - audio_waveform: np.ndarray, shape (samples,), dtype float32, range [-1, 1]
        - sample_rate: int, always TARGET_SAMPLE_RATE (16000 Hz)
        
    Raises:
        AudioFormatError: If file format unsupported or corrupted
        AudioDurationError: If audio duration invalid
        AudioLoadingError: If loading fails for any reason
    """
    # Validate format
    validate_audio_format(audio_path)
    
    try:
        # Load with auto-resampling to target sample rate
        logger.info(f"Loading audio: {audio_path}")
        waveform, sr_loaded = librosa.load(
            audio_path,
            sr=sr,
            mono=mono,
            offset=0.0,
            duration=None
        )
        
        # Validate duration
        duration_seconds = len(waveform) / sr
        
        if duration_seconds < MIN_AUDIO_DURATION_SEC:
            raise AudioDurationError(
                f"Audio too short: {duration_seconds:.1f}s (minimum: {MIN_AUDIO_DURATION_SEC}s)"
            )
        
        if duration_seconds > MAX_AUDIO_DURATION_SEC:
            raise AudioDurationError(
                f"Audio too long: {duration_seconds:.1f}s (maximum: {MAX_AUDIO_DURATION_SEC}s)"
            )
        
        # Log duration info
        duration_min = duration_seconds / 60
        logger.info(f"Audio loaded successfully: {duration_min:.1f} minutes, {sr} Hz, {waveform.shape[0]} samples")
        
        # If duration far from typical, warn
        if duration_seconds < TYPICAL_CALL_DURATION_MIN * 0.5:
            logger.warning(f"Audio much shorter than typical earnings call ({duration_min:.1f} min)")
        
        # Ensure float32 dtype
        waveform = waveform.astype(np.float32)
        
        # Verify normalization
        max_val = np.max(np.abs(waveform))
        if max_val > 1.0:
            logger.warning(f"Waveform contains values > 1.0 (max: {max_val:.4f}) - normalizing")
            waveform = waveform / (max_val + 1e-8)
        
        return waveform, sr
    
    except librosa.LibrosaError as e:
        raise AudioLoadingError(f"Librosa loading error: {str(e)}")
    except Exception as e:
        raise AudioLoadingError(f"Unexpected error loading audio: {str(e)}")


def detect_audio_corruption(
    waveform: np.ndarray,
    sr: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Detect potential audio issues before processing.
    
    Checks for:
    - Clipping (values at exactly ±1.0)
    - Silence or very quiet audio
    - Dynamic range
    - DC offset
    
    Args:
        waveform: Audio waveform array
        sr: Sample rate
        verbose: Print warnings if True
        
    Returns:
        Dictionary with detection results:
        {
            'is_corrupted': bool,
            'clipping_detected': bool,
            'clipping_percentage': float,
            'silence_ratio': float,
            'dynamic_range_db': float,
            'dc_offset': float,
            'rms_energy': float,
            'warnings': List[str],
            'quality_flag': str  # 'GOOD' | 'MARGINAL' | 'POOR'
        }
    """
    warnings_list = []
    quality_flag = 'GOOD'
    
    # 1. Clipping Detection
    # Count samples exactly at ±1.0 (indicates hard clipping)
    clipped_samples = np.sum(np.abs(waveform) >= 0.99)
    clipping_percentage = (clipped_samples / len(waveform)) * 100
    clipping_detected = clipping_percentage > 0.5  # >0.5% clipping
    
    if clipping_detected:
        warnings_list.append(
            f"Audio clipping detected: {clipping_percentage:.3f}% samples at ±0.99"
        )
        quality_flag = 'MARGINAL'
    
    # 2. RMS Energy (loudness)
    rms = np.sqrt(np.mean(waveform**2))
    
    if rms < 0.001:
        warnings_list.append(f"Very quiet audio (RMS < 0.001)")
        quality_flag = 'POOR'
    elif rms < 0.01:
        warnings_list.append(f"Quiet audio (RMS < 0.01)")
        quality_flag = 'MARGINAL'
    
    # 3. Dynamic Range
    # Remove silent frames for calculation
    signal_rms_db = 20 * np.log10(rms + 1e-8)
    
    # Find noise floor (lowest 10th percentile)
    magnitude = np.abs(waveform)
    noise_level = np.percentile(magnitude, 10)
    noise_rms_db = 20 * np.log10(noise_level + 1e-8)
    
    dynamic_range_db = signal_rms_db - noise_rms_db
    
    if dynamic_range_db < 10:
        warnings_list.append(f"Low dynamic range: {dynamic_range_db:.1f} dB (expect >10 dB)")
        quality_flag = 'MARGINAL'
    
    # 4. Silence Ratio (very low energy frames)
    # Count frames with energy < -40 dB (relative to RMS)
    frame_length = int(sr * 0.020)  # 20ms frames
    num_frames = len(waveform) // frame_length
    silence_count = 0
    
    for i in range(num_frames):
        frame = waveform[i*frame_length:(i+1)*frame_length]
        frame_rms = np.sqrt(np.mean(frame**2))
        frame_rms_db = 20 * np.log10(frame_rms + 1e-8)
        
        if frame_rms_db < -40:
            silence_count += 1
    
    silence_ratio = silence_count / max(num_frames, 1)
    
    if silence_ratio > 0.5:
        warnings_list.append(f"High silence ratio: {silence_ratio*100:.1f}% (<-40dB)")
        quality_flag = 'MARGINAL'
    
    # 5. DC Offset
    dc_offset = np.mean(waveform)
    
    if np.abs(dc_offset) > 0.05:
        warnings_list.append(f"Significant DC offset: {dc_offset:.4f}")
        quality_flag = 'MARGINAL'
    
    # 6. Check for NaN or Inf
    if np.isnan(waveform).any() or np.isinf(waveform).any():
        warnings_list.insert(0, "Audio contains NaN or Inf values!")
        quality_flag = 'POOR'
    
    result = {
        'is_corrupted': quality_flag == 'POOR',
        'clipping_detected': clipping_detected,
        'clipping_percentage': float(clipping_percentage),
        'silence_ratio': float(silence_ratio),
        'dynamic_range_db': float(dynamic_range_db),
        'dc_offset': float(dc_offset),
        'rms_energy': float(rms),
        'warnings': warnings_list,
        'quality_flag': quality_flag
    }
    
    if verbose and warnings_list:
        logger.warning(f"Audio quality issues detected ({quality_flag}):")
        for warning in warnings_list:
            logger.warning(f"  - {warning}")
    
    return result


def normalize_audio_amplitude(
    waveform: np.ndarray,
    target_rms: float = 0.1
) -> np.ndarray:
    """
    Normalize audio amplitude to target RMS level.
    
    Useful for consistent loudness across different recordings.
    
    Args:
        waveform: Audio waveform
        target_rms: Target RMS level (default: 0.1, typical range 0.05-0.15)
        
    Returns:
        Normalized waveform
    """
    current_rms = np.sqrt(np.mean(waveform**2))
    
    if current_rms < 1e-8:
        logger.warning("Waveform has zero energy, not normalizing")
        return waveform
    
    scaling_factor = target_rms / current_rms
    normalized = waveform * scaling_factor
    
    # Clip to [-1, 1] if needed
    if np.max(np.abs(normalized)) > 1.0:
        logger.warning("Normalized audio exceeds ±1.0, clipping to [-1, 1]")
        normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized.astype(np.float32)


def save_standardized_audio(
    waveform: np.ndarray,
    sr: int,
    output_path: str
) -> None:
    """
    Save standardized audio to WAV file.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        output_path: Path to save WAV file
    """
    import soundfile as sf
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(output_path, waveform, sr)
    logger.info(f"Saved standardized audio to: {output_path}")


# ============================================================================
# Convenience Functions
# ============================================================================

def load_audio_simple(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Simple wrapper for basic audio loading (16kHz mono, float32).
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        (waveform, sample_rate) tuple
    """
    return load_and_standardize_audio(audio_path)


def batch_load_audio_files(
    audio_dir: str,
    extensions: Optional[list] = None
) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Load all audio files from a directory.
    
    Args:
        audio_dir: Directory containing audio files
        extensions: List of extensions to load (default: all supported)
        
    Returns:
        Dictionary mapping filename to (waveform, sr) tuple
    """
    if extensions is None:
        extensions = list(SUPPORTED_AUDIO_FORMATS.keys())
    
    audio_dir_path = Path(audio_dir)
    loaded_files = {}
    errors = {}
    
    logger.info(f"Loading audio files from: {audio_dir}")
    
    for ext in extensions:
        for audio_file in audio_dir_path.glob(f"*{ext}"):
            try:
                waveform, sr = load_and_standardize_audio(str(audio_file))
                loaded_files[audio_file.name] = (waveform, sr)
                logger.info(f"  ✓ Loaded: {audio_file.name}")
            except Exception as e:
                errors[audio_file.name] = str(e)
                logger.error(f"  ✗ Failed to load {audio_file.name}: {str(e)}")
    
    logger.info(f"Loaded {len(loaded_files)} files, {len(errors)} errors")
    
    if errors:
        logger.warning("Failed files:")
        for filename, error in errors.items():
            logger.warning(f"  - {filename}: {error}")
    
    return loaded_files


if __name__ == "__main__":
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_loader.py <audio_file>")
        print("\nExample: python audio_loader.py earnings_call.mp3")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        # Load audio
        print(f"\n📻 Loading: {audio_file}")
        waveform, sr = load_and_standardize_audio(audio_file)
        print(f"✓ Loaded successfully!")
        print(f"  - Duration: {len(waveform)/sr/60:.1f} minutes")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - Channels: 1 (mono)")
        print(f"  - Data type: {waveform.dtype}")
        
        # Check for corruption
        print(f"\n🔍 Checking audio quality...")
        quality = detect_audio_corruption(waveform, sr)
        print(f"  - Quality flag: {quality['quality_flag']}")
        print(f"  - RMS energy: {quality['rms_energy']:.4f}")
        print(f"  - Dynamic range: {quality['dynamic_range_db']:.1f} dB")
        print(f"  - Silence ratio: {quality['silence_ratio']*100:.1f}%")
        print(f"  - Clipping: {quality['clipping_percentage']:.3f}%")
        
        if quality['warnings']:
            print(f"\n⚠️  Warnings ({len(quality['warnings'])}):")
            for warning in quality['warnings']:
                print(f"    - {warning}")
        
        print("\n✅ Ready for feature extraction!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
