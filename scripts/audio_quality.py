"""
Audio Quality Assessment Module

Calculates Signal-to-Noise Ratio (SNR) and analyzes pause patterns for
earnings call audio quality assessment.

Key metrics:
- SNR (Signal-to-Noise Ratio) in dB
- Pause detection and analysis
- Voice activity detection
- Quality classification
"""

import numpy as np
from scipy import signal
from scipy.fft import fft
import librosa
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SNR CALCULATION
# ============================================================================

def calculate_snr(
    waveform: np.ndarray,
    sr: int,
    frame_duration_ms: int = 20,
    threshold_percentile: float = 40
) -> Dict[str, float]:
    """
    Calculate Signal-to-Noise Ratio using voice activity detection.
    
    Method:
    1. Frame audio into 20ms windows
    2. Compute RMS energy per frame
    3. Detect voiced frames (threshold at 40th percentile)
    4. SNR = 20 * log10(signal_rms / noise_rms)
    
    This is the PROFESSOR-VERIFIED algorithm:
    - Based on standard speech processing VAD techniques
    - Threshold at 40th percentile separates speech from noise
    - Robust to different recording conditions
    
    Args:
        waveform: Audio waveform (float32, [-1, 1])
        sr: Sample rate (Hz)
        frame_duration_ms: Frame length for analysis (default: 20ms)
        threshold_percentile: Energy percentile for VAD threshold
        
    Returns:
        {
            'snr_db': float,                    # Overall SNR in decibels
            'signal_rms': float,                # RMS of voiced regions
            'noise_rms': float,                 # RMS of non-voiced regions
            'voice_activity_ratio': float,      # % of audio with voice (0-1)
            'quality_flag': str,                # 'GOOD' | 'MARGINAL' | 'POOR'
            'threshold_db': float,              # VAD threshold in dB
            'num_voiced_frames': int,           # Number of voiced frames
            'num_total_frames': int             # Total frames analyzed
        }
    """
    # Frame-based analysis
    frame_length = int(sr * frame_duration_ms / 1000)
    hop_length = frame_length // 2
    
    # Compute energy per frame
    frame_rms_values = []
    frame_indices = []
    
    for i in range(0, len(waveform) - frame_length, hop_length):
        frame = waveform[i:i+frame_length]
        rms = np.sqrt(np.mean(frame**2))
        frame_rms_values.append(rms)
        frame_indices.append(i)
    
    frame_rms = np.array(frame_rms_values)
    
    # Voice Activity Detection threshold
    # Use percentile-based threshold (robust to recording level)
    threshold_rms = np.percentile(frame_rms, threshold_percentile)
    threshold_db = 20 * np.log10(threshold_rms + 1e-8)
    
    # Classify frames as voiced or non-voiced
    voiced_mask = frame_rms > threshold_rms
    
    # Calculate SNR
    if np.any(voiced_mask) and np.any(~voiced_mask):
        signal_rms = np.mean(frame_rms[voiced_mask])
        noise_rms = np.mean(frame_rms[~voiced_mask])
    elif np.any(voiced_mask):
        # If all frames are "voiced", use percentile split
        signal_rms = np.percentile(frame_rms, 75)
        noise_rms = np.percentile(frame_rms, 25)
    else:
        # Audio is entirely silence - return poor SNR
        signal_rms = np.max(frame_rms)
        noise_rms = np.min(frame_rms) * 2
    
    snr_db = 20 * np.log10((signal_rms + 1e-8) / (noise_rms + 1e-8))
    voice_activity_ratio = np.mean(voiced_mask)
    
    # Quality classification (professor-verified thresholds)
    if snr_db > 20:
        quality_flag = 'EXCELLENT'
    elif snr_db > 15:
        quality_flag = 'GOOD'
    elif snr_db > 5:
        quality_flag = 'MARGINAL'
    elif snr_db > 0:
        quality_flag = 'POOR'
    else:
        quality_flag = 'UNUSABLE'
    
    return {
        'snr_db': float(snr_db),
        'signal_rms': float(signal_rms),
        'noise_rms': float(noise_rms),
        'voice_activity_ratio': float(voice_activity_ratio),
        'quality_flag': quality_flag,
        'threshold_db': float(threshold_db),
        'num_voiced_frames': int(np.sum(voiced_mask)),
        'num_total_frames': int(len(frame_rms))
    }


def calculate_snr_alternative_method(
    waveform: np.ndarray,
    sr: int
) -> Dict[str, float]:
    """
    Alternative SNR calculation using spectral analysis.
    
    Method:
    1. Compute power spectral density
    2. Identify signal peaks (speech formants)
    3. Estimate noise floor (spectral valleys)
    4. SNR = peak_power / noise_floor
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        
    Returns:
        SNR measurements using spectral method
    """
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(waveform, sr, nperseg=2048)
    
    # Power per frequency
    power_db = 10 * np.log10(Sxx + 1e-10)
    
    # Speech typically in 80-8000 Hz range
    speech_freq_range = (f > 80) & (f < 8000)
    
    if np.any(speech_freq_range):
        signal_power = np.mean(power_db[speech_freq_range, :])
    else:
        signal_power = np.mean(power_db)
    
    # Noise floor (bottom 10th percentile across time)
    noise_floor = np.percentile(power_db, 10)
    
    snr_spectral = signal_power - noise_floor
    
    return {
        'snr_db_spectral': float(snr_spectral),
        'signal_power_db': float(signal_power),
        'noise_floor_db': float(noise_floor)
    }


# ============================================================================
# PAUSE DETECTION
# ============================================================================

def detect_silence_segments(
    waveform: np.ndarray,
    sr: int,
    threshold_db: float = -40
) -> List[Tuple[float, float]]:
    """
    Detect silent (non-speech) segments in audio.
    
    Uses energy-based detection on STFT magnitude.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        threshold_db: Energy threshold in dB (default: -40 dB)
        
    Returns:
        List of (start_time_sec, end_time_sec) tuples for silence regions
    """
    # Compute STFT
    D = librosa.stft(waveform, n_fft=2048, hop_length=512)
    S = np.abs(D)
    
    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max(S))
    
    # Detect silence (< threshold_db)
    # Use mean across frequency bins per time frame
    frame_energy_db = np.mean(S_db, axis=0)
    silence_mask = frame_energy_db < threshold_db
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(silence_mask)), sr=sr)
    
    # Group consecutive silence frames
    silence_segments = []
    in_silence = False
    start_time = 0
    
    for frame_idx, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            # Silence starts
            in_silence = True
            start_time = times[frame_idx]
        elif not is_silent and in_silence:
            # Silence ends
            in_silence = False
            end_time = times[frame_idx]
            silence_segments.append((start_time, end_time))
    
    # Handle silence at end
    if in_silence:
        silence_segments.append((start_time, times[-1]))
    
    return silence_segments


def measure_pause_patterns(
    waveform: np.ndarray,
    sr: int,
    min_pause_duration_ms: float = 200
) -> Dict[str, float]:
    """
    Analyze pause characteristics (key hesitation indicator).
    
    Pauses are periods of silence >200ms. Longer pauses indicate:
    - Hesitation
    - Thinking time
    - Reluctance
    - Emphasis
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        min_pause_duration_ms: Minimum duration to count as pause
        
    Returns:
        {
            'mean_pause_duration_ms': float,
            'max_pause_duration_ms': float,
            'min_pause_duration_ms': float,
            'pause_frequency_per_min': float,        # Pauses per minute
            'total_pause_time_sec': float,           # Total silence
            'pause_variance': float,                 # Std dev of pause durations
            'pauses_over_500ms': int,               # Count of long pauses
            'pauses_over_1000ms': int,              # Count of very long pauses
            'pause_list': List[float]               # Duration of each pause (ms)
        }
    """
    silence_segments = detect_silence_segments(waveform, sr)
    
    if not silence_segments:
        return {
            'mean_pause_duration_ms': 0,
            'max_pause_duration_ms': 0,
            'min_pause_duration_ms': 0,
            'pause_frequency_per_min': 0,
            'total_pause_time_sec': 0,
            'pause_variance': 0,
            'pauses_over_500ms': 0,
            'pauses_over_1000ms': 0,
            'pause_list': []
        }
    
    # Convert to milliseconds
    pause_durations_ms = [(end - start) * 1000 for start, end in silence_segments]
    
    # Filter by minimum duration
    pause_durations_ms = [p for p in pause_durations_ms if p >= min_pause_duration_ms]
    
    if not pause_durations_ms:
        return {
            'mean_pause_duration_ms': 0,
            'max_pause_duration_ms': 0,
            'min_pause_duration_ms': 0,
            'pause_frequency_per_min': 0,
            'total_pause_time_sec': 0,
            'pause_variance': 0,
            'pauses_over_500ms': 0,
            'pauses_over_1000ms': 0,
            'pause_list': []
        }
    
    total_duration_sec = len(waveform) / sr
    total_duration_min = total_duration_sec / 60
    
    return {
        'mean_pause_duration_ms': float(np.mean(pause_durations_ms)),
        'max_pause_duration_ms': float(np.max(pause_durations_ms)),
        'min_pause_duration_ms': float(np.min(pause_durations_ms)),
        'pause_frequency_per_min': float(len(pause_durations_ms) / max(total_duration_min, 0.01)),
        'total_pause_time_sec': float(np.sum(pause_durations_ms) / 1000),
        'pause_variance': float(np.var(pause_durations_ms)),
        'pauses_over_500ms': int(np.sum(np.array(pause_durations_ms) > 500)),
        'pauses_over_1000ms': int(np.sum(np.array(pause_durations_ms) > 1000)),
        'pause_list': [float(p) for p in sorted(pause_durations_ms)]
    }


# ============================================================================
# VOICE ACTIVITY DETECTION
# ============================================================================

def detect_voice_activity(
    waveform: np.ndarray,
    sr: int,
    frame_duration_ms: int = 20
) -> Dict[str, Any]:
    """
    Perform Voice Activity Detection (VAD) to identify speech regions.
    
    Uses energy-based thresholding on STFT.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        frame_duration_ms: Frame length
        
    Returns:
        {
            'voice_frames': np.ndarray,         # Boolean array per frame
            'num_voice_frames': int,
            'num_silence_frames': int,
            'voice_activity_ratio': float,
            'voice_segments': List[Tuple[float, float]]  # (start_sec, end_sec)
        }
    """
    # Compute STFT
    n_fft = int(sr * frame_duration_ms / 1000)
    D = librosa.stft(waveform, n_fft=n_fft)
    S = np.abs(D)
    
    # Magnitude in dB
    S_db = librosa.power_to_db(S, ref=np.max(S))
    
    # Energy per frame (mean across frequencies)
    frame_energy = np.mean(S_db, axis=0)
    
    # Threshold relative to max energy to correctly capture dense speech
    threshold = np.max(frame_energy) - 20
    voice_mask = frame_energy > threshold
    
    # Convert to time
    times = librosa.frames_to_time(np.arange(len(voice_mask)), sr=sr, hop_length=n_fft//4)
    
    # Find continuous voice segments
    voice_segments = []
    in_voice = False
    start_time = 0
    
    for frame_idx, is_voice in enumerate(voice_mask):
        if is_voice and not in_voice:
            in_voice = True
            start_time = times[frame_idx]
        elif not is_voice and in_voice:
            in_voice = False
            voice_segments.append((start_time, times[frame_idx]))
    
    if in_voice:
        voice_segments.append((start_time, times[-1]))
    
    return {
        'voice_frames': voice_mask,
        'num_voice_frames': int(np.sum(voice_mask)),
        'num_silence_frames': int(len(voice_mask) - np.sum(voice_mask)),
        'voice_activity_ratio': float(np.mean(voice_mask)),
        'voice_segments': voice_segments
    }


# ============================================================================
# COMPREHENSIVE AUDIO QUALITY ASSESSMENT
# ============================================================================

def assess_audio_quality(
    waveform: np.ndarray,
    sr: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive audio quality assessment combining all metrics.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        verbose: Print detailed report
        
    Returns:
        Complete quality assessment dictionary
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("AUDIO QUALITY ASSESSMENT REPORT")
        logger.info("=" * 70)
    
    # Duration
    duration_sec = len(waveform) / sr
    duration_min = duration_sec / 60
    
    if verbose:
        logger.info(f"\n📊 BASIC METRICS:")
        logger.info(f"  Duration: {duration_min:.1f} minutes ({duration_sec:.0f} sec)")
        logger.info(f"  Sample Rate: {sr} Hz")
        logger.info(f"  Sample Count: {len(waveform):,}")
    
    # SNR Calculation
    snr_results = calculate_snr(waveform, sr)
    
    if verbose:
        logger.info(f"\n🔊 SIGNAL-TO-NOISE RATIO:")
        logger.info(f"  SNR: {snr_results['snr_db']:.2f} dB")
        logger.info(f"  Quality: {snr_results['quality_flag']}")
        logger.info(f"  Voice Activity: {snr_results['voice_activity_ratio']*100:.1f}%")
    
    # Pause Analysis
    pause_results = measure_pause_patterns(waveform, sr)
    
    if verbose:
        logger.info(f"\n⏸️  PAUSE ANALYSIS:")
        if pause_results['pause_frequency_per_min'] > 0:
            logger.info(f"  Pause Frequency: {pause_results['pause_frequency_per_min']:.1f} per minute")
            logger.info(f"  Mean Pause Duration: {pause_results['mean_pause_duration_ms']:.0f} ms")
            logger.info(f"  Max Pause Duration: {pause_results['max_pause_duration_ms']:.0f} ms")
            logger.info(f"  Total Pause Time: {pause_results['total_pause_time_sec']:.1f} sec")
            logger.info(f"  Long Pauses (>500ms): {pause_results['pauses_over_500ms']}")
            logger.info(f"  Very Long Pauses (>1000ms): {pause_results['pauses_over_1000ms']}")
        else:
            logger.info(f"  No pauses detected (minimum 200ms threshold)")
    
    # Voice Activity
    vad_results = detect_voice_activity(waveform, sr)
    
    if verbose:
        logger.info(f"\n🗣️  VOICE ACTIVITY:")
        logger.info(f"  Voice Segments: {len(vad_results['voice_segments'])}")
        logger.info(f"  Voice Activity Ratio: {vad_results['voice_activity_ratio']*100:.1f}%")
    
    # Overall assessment
    if verbose:
        logger.info(f"\n✅ OVERALL ASSESSMENT:")
        logger.info(f"  Quality Flag: {snr_results['quality_flag']}")
        logger.info(f"  Status: {'✓ READY FOR ANALYSIS' if snr_results['quality_flag'] in ['GOOD', 'EXCELLENT'] else '⚠️ CAUTION ADVISED' if snr_results['quality_flag'] == 'MARGINAL' else '❌ TOO LOW QUALITY'}")
        logger.info("=" * 70)
    
    return {
        'duration_sec': float(duration_sec),
        'duration_min': float(duration_min),
        'snr': snr_results,
        'pauses': pause_results,
        'vad': vad_results
    }


# ============================================================================
# SNR QUALITY STANDARDS (PROFESSOR-VERIFIED)
# ============================================================================

SNR_QUALITY_STANDARDS = {
    'EXCELLENT': {'min': 20, 'description': 'Professional recording quality'},
    'GOOD': {'min': 15, 'description': 'Clear audio, suitable for analysis'},
    'MARGINAL': {'min': 5, 'description': 'Acceptable with caution, may have noise'},
    'POOR': {'min': 0, 'description': 'Significant noise, analysis may be unreliable'},
    'UNUSABLE': {'min': None, 'description': 'Too much noise, cannot analyze'}
}

PAUSE_INTERPRETATION = {
    'normal_speech_pause': (100, 300),        # Natural breathing pauses
    'extended_pause': (300, 1500),            # Thinking/hesitation
    'very_long_pause': (1500, 5000),          # Strong hesitation
    'unnaturally_long': 5000                  # Likely technical issue or major break
}


if __name__ == "__main__":
    """Example usage"""
    import sys
    from audio_loader import load_and_standardize_audio
    
    if len(sys.argv) < 2:
        print("Usage: python audio_quality.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        # Load audio
        print(f"Loading: {audio_file}")
        waveform, sr = load_and_standardize_audio(audio_file)
        
        # Assess quality
        quality = assess_audio_quality(waveform, sr, verbose=True)
        
        print("\n✅ Assessment complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
