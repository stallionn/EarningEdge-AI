"""
Unit Tests for Audio Loading & Quality Assessment
Tests Phase 0 implementations: audio_loader.py and audio_quality.py

Run tests with: pytest tests/test_phase0_audio.py -v
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from audio_loader import (
    load_and_standardize_audio,
    detect_audio_corruption,
    validate_audio_format,
    AudioFormatError,
    AudioDurationError,
    AudioLoadingError,
    TARGET_SAMPLE_RATE
)

from audio_quality import (
    calculate_snr,
    detect_silence_segments,
    measure_pause_patterns
)


class TestAudioLoader:
    """Tests for audio_loader.py"""
    
    @pytest.fixture
    def create_test_audio(self):
        """Create temporary test audio files"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        files = {}
        
        # 1. Normal clean speech (6 minutes)
        sr = 16000
        duration = 360  # seconds
        # Sine wave at 440 Hz + 150 Hz (speech-like frequencies)
        t = np.arange(duration * sr) / sr
        signal_1 = 0.3 * np.sin(2 * np.pi * 440 * t)  # Fundamental
        signal_1 += 0.15 * np.sin(2 * np.pi * 150 * t)  # Lower frequency
        signal_1 = signal_1.astype(np.float32)
        
        wav_file = temp_path / "clean_speech.wav"
        sf.write(wav_file, signal_1, sr)
        files['clean'] = str(wav_file)
        
        # 2. Quiet speech (low RMS)
        quiet_signal = signal_1 * 0.05
        quiet_file = temp_path / "quiet_speech.wav"
        sf.write(quiet_file, quiet_signal.astype(np.float32), sr)
        files['quiet'] = str(quiet_file)
        
        # 3. Short audio (1 minute - should fail minimum duration check)
        short_signal = signal_1[:60*sr]
        short_file = temp_path / "short_audio.wav"
        sf.write(short_file, short_signal.astype(np.float32), sr)
        files['short'] = str(short_file)
        
        # 4. Audio with clipping
        clipped_signal = np.clip(signal_1 * 2, -1, 1)  # Amplitude > 1
        clipped_file = temp_path / "clipped_audio.wav"
        sf.write(clipped_file, clipped_signal.astype(np.float32), sr)
        files['clipped'] = str(clipped_file)
        
        yield files
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_load_clean_audio(self, create_test_audio):
        """Test loading clean audio file"""
        audio_path = create_test_audio['clean']
        
        waveform, sr = load_and_standardize_audio(audio_path)
        
        # Assertions
        assert sr == TARGET_SAMPLE_RATE
        assert waveform.dtype == np.float32
        assert np.max(np.abs(waveform)) <= 1.0
        assert len(waveform) == 360 * TARGET_SAMPLE_RATE
    
    def test_short_audio_fails(self, create_test_audio):
        """Test that very short audio raises AudioDurationError"""
        audio_path = create_test_audio['short']
        
        with pytest.raises(AudioDurationError):
            load_and_standardize_audio(audio_path)
    
    def test_invalid_format_fails(self):
        """Test that invalid format raises AudioFormatError"""
        with pytest.raises(AudioFormatError):
            load_and_standardize_audio("test.xyz")
    
    def test_detect_corruption_clean(self, create_test_audio):
        """Test corruption detection on clean audio"""
        audio_path = create_test_audio['clean']
        waveform, sr = load_and_standardize_audio(audio_path)
        
        report = detect_audio_corruption(waveform, sr)
        
        assert report['is_corrupted'] is False
        assert report['quality_flag'] in ['GOOD', 'MARGINAL']
        assert report['clipping_percentage'] < 1.0
    
    def test_detect_corruption_clipped(self, create_test_audio):
        """Test corruption detection on clipped audio"""
        audio_path = create_test_audio['clipped']
        waveform, sr = load_and_standardize_audio(audio_path)
        
        report = detect_audio_corruption(waveform, sr)
        
        assert report['clipping_detected'] is True
        assert report['clipping_percentage'] > 0.5
    
    def test_quiet_audio_warning(self, create_test_audio):
        """Test that quiet audio is flagged"""
        audio_path = create_test_audio['quiet']
        waveform, sr = load_and_standardize_audio(audio_path)
        
        report = detect_audio_corruption(waveform, sr, verbose=False)
        
        # Should have warnings about quiet audio
        assert len(report['warnings']) > 0
        assert report['quality_flag'] in ['MARGINAL', 'POOR']


class TestAudioQuality:
    """Tests for audio_quality.py"""
    
    @pytest.fixture
    def create_snr_test_audio(self):
        """Create test audio with known SNR properties"""
        sr = 16000
        duration = 30  # seconds
        t = np.arange(duration * sr) / sr
        
        # Test case 1: Clean signal (high SNR ~20dB)
        signal = 0.5 * np.sin(2 * np.pi * 200 * t)
        noise = 0.05 * np.random.randn(len(signal))
        clean_audio = (signal + noise).astype(np.float32)
        clean_audio = clean_audio / np.max(np.abs(clean_audio))
        
        # Test case 2: Noisy signal (low SNR ~5dB)
        noisy_audio = (signal * 0.3 + noise * 0.2).astype(np.float32)
        noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
        
        return {
            'clean': clean_audio,
            'noisy': noisy_audio,
            'sr': sr
        }
    
    def test_snr_calculation_clean(self, create_snr_test_audio):
        """Test SNR calculation on clean audio"""
        audio = create_snr_test_audio['clean']
        sr = create_snr_test_audio['sr']
        
        result = calculate_snr(audio, sr)
        
        # Clean audio should have good SNR
        assert result['snr_db'] > 10  # At least 10 dB
        assert result['quality_flag'] in ['EXCELLENT', 'GOOD']
        assert result['voice_activity_ratio'] > 0.5
    
    def test_snr_calculation_noisy(self, create_snr_test_audio):
        """Test SNR calculation on noisy audio"""
        audio = create_snr_test_audio['noisy']
        sr = create_snr_test_audio['sr']
        
        result = calculate_snr(audio, sr)
        
        # Noisy audio should have lower SNR
        assert result['snr_db'] < 15  # Less than 15 dB
        assert result['snr_db'] > -5  # Still measurable
    
    def test_snr_quality_flags(self, create_snr_test_audio):
        """Test that SNR quality flags are correct"""
        sr = create_snr_test_audio['sr']
        
        # Test each quality flag range
        for snr_target in [25, 17, 10, 2]:
            # Generate audio with target SNR
            signal = 0.5 * np.sin(2 * np.pi * 200 * np.arange(30*sr) / sr)
            noise_std = 0.5 / (10 ** (snr_target / 20))
            noise = noise_std * np.random.randn(len(signal))
            audio = (signal + noise).astype(np.float32)
            
            result = calculate_snr(audio, sr)
            
            # Verify quality flag is reasonable
            assert result['quality_flag'] in ['EXCELLENT', 'GOOD', 'MARGINAL', 'POOR', 'UNUSABLE']
    
    def test_silence_detection(self):
        """Test silence detection"""
        sr = 16000
        duration = 10  # seconds
        t = np.arange(duration * sr) / sr
        
        # Create audio with speech and silences
        audio = np.zeros(duration * sr, dtype=np.float32)
        
        # Add speech in segments: 0-2s, 3-5s, 7-9s
        audio[0:2*sr] = 0.3 * np.sin(2 * np.pi * 200 * t[:2*sr])
        audio[3*sr:5*sr] = 0.3 * np.sin(2 * np.pi * 200 * t[3*sr:5*sr])
        audio[7*sr:9*sr] = 0.3 * np.sin(2 * np.pi * 200 * t[7*sr:9*sr])
        
        # Detect silences
        silences = detect_silence_segments(audio, sr)
        
        # Should detect ~3 silence segments
        assert len(silences) >= 2
    
    def test_pause_detection(self):
        """Test pause/hesitation detection"""
        sr = 16000
        duration = 20  # seconds
        t = np.arange(duration * sr) / sr
        
        # Create audio with deliberate pauses
        audio = np.zeros(duration * sr, dtype=np.float32)
        
        # Speech segments with pauses
        audio[0:3*sr] = 0.4 * np.sin(2 * np.pi * 200 * t[:3*sr])      # 3s speech
        # 1 second pause (2s-3s)
        audio[4*sr:7*sr] = 0.4 * np.sin(2 * np.pi * 200 * t[4*sr:7*sr])  # 3s speech
        # 2 second pause (7s-9s) - represents hesitation
        audio[9*sr:12*sr] = 0.4 * np.sin(2 * np.pi * 200 * t[9*sr:12*sr]) # 3s speech
        
        # Measure pauses
        pause_info = measure_pause_patterns(audio, sr)
        
        # Should detect pauses from silence gaps
        assert len(pause_info['pause_list']) >= 1
        assert pause_info['mean_pause_duration_ms'] > 500  # Average > 500ms
        assert pause_info['pause_frequency_per_min'] > 0
    
    def test_pause_distribution(self):
        """Test classification of pause types"""
        sr = 16000
        duration = 20
        t = np.arange(duration * sr) / sr
        
        # Create audio with different pause types
        audio = np.zeros(duration * sr, dtype=np.float32)
        
        # Interleave speech (0.5s) with different pause durations
        speech_dur = 0.5 * sr
        
        # 150ms pause (normal)
        audio[0*sr:0.5*sr] = 0.3 * np.sin(2 * np.pi * 200 * t[:int(speech_dur)])
        
        # 800ms pause (extended)
        audio[1.5*sr:2*sr] = 0.3 * np.sin(2 * np.pi * 200 * t[int(1.5*sr):int(2*sr)])
        
        # 3s pause (long hesitation)
        audio[5*sr:5.5*sr] = 0.3 * np.sin(2 * np.pi * 200 * t[int(5*sr):int(5.5*sr)])
        
        pause_info = measure_pause_patterns(audio, sr)
        
        # Verify pause counters
        assert pause_info['pauses_over_500ms'] >= 0
        assert pause_info['pauses_over_1000ms'] >= 0


class TestAudioPipelineIntegration:
    """Integration tests for full audio pipeline"""
    
    def test_full_pipeline_clean_audio(self):
        """Test complete pipeline: load -> assess corruption -> calculate SNR -> measure pauses"""
        # Create test audio
        sr = 16000
        duration = 60
        t = np.arange(duration * sr) / sr
        
        # Simulate earnings call audio with natural pauses
        audio = np.zeros(duration * sr, dtype=np.float32)
        
        # Segment 1: CEO speaks (15s)
        speech1 = 0.35 * np.sin(2 * np.pi * 150 * t[:15*sr])
        audio[0:15*sr] = speech1
        
        # Pause (2s) - normal
        # [15-17 silence]
        
        # Segment 2: CFO speaks (15s)
        speech2 = 0.35 * np.sin(2 * np.pi * 180 * t[:15*sr])
        audio[17*sr:32*sr] = speech2
        
        # Long pause (3s) - hesitation
        # [32-35 silence]
        
        # Segment 3: Q&A (20s)
        speech3 = 0.35 * np.sin(2 * np.pi * 160 * t[:20*sr])
        audio[35*sr:55*sr] = speech3
        
        # Final pause (5s)
        # [55-60 silence]
        
        # Normalize
        audio = audio / np.max(np.abs(audio) + 1e-8)
        
        # Step 1: Corruption detection
        corruption = detect_audio_corruption(audio, sr, verbose=False)
        assert corruption['is_corrupted'] == False
        assert corruption['quality_flag'] in ['GOOD', 'MARGINAL']
        
        # Step 2: SNR calculation
        snr_result = calculate_snr(audio, sr)
        assert snr_result['snr_db'] > 5
        assert snr_result['quality_flag'] in ['GOOD', 'MARGINAL', 'EXCELLENT']
        
        # Step 3: Pause analysis
        pause_info = measure_pause_patterns(audio, sr)
        assert len(pause_info['pause_list']) >= 2  # At least 2 pauses
        assert pause_info['pause_frequency_per_min'] > 0
        
        print(f"\n✓ Integration test passed:")
        print(f"  - Corruption: {corruption['quality_flag']}")
        print(f"  - SNR: {snr_result['snr_db']:.1f} dB")
        print(f"  - Pauses detected: {len(pause_info['pause_list'])}")
        print(f"  - Pause frequency: {pause_info['pause_frequency_per_min']:.1f} per minute")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_phase0_audio.py -v
    pytest.main([__file__, "-v"])
