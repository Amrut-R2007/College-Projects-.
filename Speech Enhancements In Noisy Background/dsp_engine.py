"""
MODULE 1 & 2: AUDIO FRONTEND, RECONSTRUCTION, & PSYCHOACOUSTIC SUBTRACTION CORE (Member 1 & 2 Domains)
Implements all NumPy/SciPy-based raw DSP code for framing, STFT, Bark mapping, spectral subtraction, and Weighted Overlap-Add (WOLA).
"""

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import convolve1d

class DSPEngine:
    def __init__(self, sample_rate=16000, frame_ms=30, hop_ms=15, num_bark_bands=24, beta=0.01):
        """
        Initializes the DSP engine with standard parameters.
        """
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * (frame_ms / 1000.0))  # 480 samples @ 16kHz
        self.hop_len = int(sample_rate * (hop_ms / 1000.0))      # 240 samples @ 16kHz (50% overlap)
        self.fft_size = self.frame_len                            # 480 points DFT
        self.num_bins = (self.fft_size // 2) + 1                  # 241 linear frequency bins
        self.num_bark_bands = num_bark_bands
        self.beta = beta                                          # Spectral floor constant
        
        # 1. Apply a Hann Window to satisfy the Constant Overlap-Add (COLA) constraint.
        # Periodic Hann window is perfect for COLA with 50% overlap
        self.analysis_window = np.hanning(self.frame_len)
        self.synthesis_window = np.hanning(self.frame_len)
        
        # 2. Build static Linear-to-Bark Transformation Matrix M_{L->B} of shape (24, 241)
        self.M_LB = self._build_bark_matrix()
        
        # 3. Pre-compute the Bark band frequency-weighting array (W_b) to target colored noise
        # Lower bands get higher weight (aggressive suppression of low-frequency hum/rumble)
        # Mid bands get standard weight, High bands get lower weight (preserves high-frequency consonant details)
        self.W_b = np.ones(self.num_bark_bands, dtype=np.float32)
        for b in range(self.num_bark_bands):
            if b < 4:
                self.W_b[b] = 2.2  # De-hum (0 to ~400 Hz) - Aggressive
            elif b < 14:
                self.W_b[b] = 1.5  # Core speech spectrum (~400 Hz to 2400 Hz) - Aggressive
            else:
                self.W_b[b] = 0.8  # Prevent speech muffling (>2400 Hz) - Aggressive
                
        # 4. Map the Bark band weights back to the 241 linear frequency bins for fast vectorization
        # Using the transpose projection or direct mapping
        self.W_linear = np.zeros(self.num_bins, dtype=np.float32)
        # Find which Bark band each FFT bin belongs to, and assign its weight
        bin_frequencies = np.linspace(0, self.sample_rate / 2.0, self.num_bins)
        for k, f in enumerate(bin_frequencies):
            bark_val = self._hz_to_bark(f)
            band_idx = self._bark_to_band_index(bark_val)
            self.W_linear[k] = self.W_b[band_idx]

    def _hz_to_bark(self, f):
        """
        Computes the standard continuous Bark scale value for a given frequency in Hz.
        Bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)^2)
        """
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    def _bark_to_band_index(self, bark):
        """
        Maps a Bark scale value to one of the 24 equally spaced Bark bands between 0 and Bark(Nyquist).
        """
        max_bark = self._hz_to_bark(self.sample_rate / 2.0) # Bark(8000) ~ 21.27
        band_width = max_bark / self.num_bark_bands
        idx = int(np.floor(bark / band_width))
        return min(self.num_bark_bands - 1, max(0, idx))

    def _build_bark_matrix(self):
        """
        Builds the static Linear-to-Bark Transformation Matrix M_{L->B} of shape (24, 241)
        """
        matrix = np.zeros((self.num_bark_bands, self.num_bins), dtype=np.float32)
        bin_frequencies = np.linspace(0, self.sample_rate / 2.0, self.num_bins)
        
        for k, f in enumerate(bin_frequencies):
            bark_val = self._hz_to_bark(f)
            band_idx = self._bark_to_band_index(bark_val)
            matrix[band_idx, k] = 1.0
            
        # Normalize each row so that it computes the AVERAGE energy within that Bark band
        # (prevents wider high-frequency bands from dominating in energy due to bin count differences)
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid divide-by-zero for empty bands (if any)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums
        return matrix

    def load_audio(self, file_path_or_buffer):
        """
        Loads standard WAV file, downmixes to mono if stereo, and normalizes to float32 range [-1.0, 1.0].
        Converts the sample rate to 16kHz if it differs.
        """
        sr, audio = wavfile.read(file_path_or_buffer)
        
        # Convert to float32 range [-1.0, 1.0]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        else:
            audio = audio.astype(np.float32)
            
        # Downmix to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Simple decimation/interpolation to 16kHz if needed
        if sr != self.sample_rate:
            # Resample using basic linear interpolation for simplicity and speed
            duration = len(audio) / sr
            num_samples = int(duration * self.sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio), num_samples),
                np.arange(len(audio)),
                audio
            )
            
        return audio

    def time_domain_framing(self, audio):
        """
        Splits a continuous time signal into 50% overlapping frames.
        Pads the signal at the end to ensure all samples are processed.
        """
        pad_len = self.frame_len - (len(audio) - self.frame_len) % self.hop_len
        padded_audio = np.pad(audio, (0, pad_len), mode='constant')
        
        num_frames = (len(padded_audio) - self.frame_len) // self.hop_len + 1
        frames = np.zeros((num_frames, self.frame_len), dtype=np.float32)
        
        for t in range(num_frames):
            start = t * self.hop_len
            frames[t] = padded_audio[start:start + self.frame_len]
            
        return frames

    def compute_stft(self, frames):
        """
        Computes the Short-Time Fourier Transform using np.fft.rfft.
        Extracts Magnitude Spectrum |Y_t(f)| and Phase Spectrum ∠Y_t(f).
        """
        num_frames = frames.shape[0]
        mags = np.zeros((num_frames, self.num_bins), dtype=np.float32)
        phases = np.zeros((num_frames, self.num_bins), dtype=np.float32)
        
        for t in range(num_frames):
            # Apply analysis Hann window
            windowed_frame = frames[t] * self.analysis_window
            # Compute RFFT
            spec = np.fft.rfft(windowed_frame, n=self.fft_size)
            mags[t] = np.abs(spec)
            phases[t] = np.angle(spec)
            
        return mags, phases

    def project_linear_to_bark(self, magnitudes):
        """
        Vectorized function to project 241 linear FFT bins onto the 24 Bark bands.
        Formula: B = M_{L->B} x |Y_t(f)|
        """
        # input magnitudes: shape (seq_len, 241)
        # matrix M_LB: shape (24, 241)
        # Output: B shape (seq_len, 24)
        return np.dot(magnitudes, self.M_LB.T)

    def spectral_subtraction_core(self, Y_mag, alpha_t, D_mag):
        """
        Implements Linear Magnitude Spectral Subtraction with Soft Attenuation (Wiener-style),
        3-point convolve frequency bin smoothing, and temporal IIR smoothing.
        """
        seq_len = Y_mag.shape[0]
        S_mag = np.zeros_like(Y_mag)
        
        # Pre-weighted linear noise magnitude profile: shape (241,)
        weighted_noise = D_mag * self.W_linear
        
        # Temporal history buffer
        prev_clean_mag = None
        
        for t in range(seq_len):
            # Calculate Wiener-style soft attenuation factor H, bounded to soft floor of 0.15
            H = np.maximum(1.0 - alpha_t[t] * (weighted_noise / (Y_mag[t] + 1e-6)), self.beta)
            clean_mag = H * Y_mag[t]
            
            # Smooth across frequency bins to prevent sharp edges (prevents phase tearing echo)
            clean_mag = convolve1d(clean_mag, weights=[0.15, 0.7, 0.15], mode='mirror')
            
            # Temporal smoothing (First-order IIR filter)
            if t == 0:
                prev_clean_mag = clean_mag.copy()
            else:
                clean_mag = 0.4 * prev_clean_mag + 0.6 * clean_mag
                prev_clean_mag = clean_mag.copy()
                
            S_mag[t] = clean_mag
            
        return S_mag

    def reconstruct_audio(self, S_mag, phases):
        """
        Implements backend reconstruction using Inverse RFFT (np.fft.irfft)
        and a Weighted Overlap-Add (WOLA) synthesis buffer with strict COLA squared window normalization.
        """
        # Ensure that clean magnitude and noisy phase shapes match exactly (anti-echo dimensions)
        assert S_mag.shape == phases.shape, f"Dimension mismatch! Clean mag shape {S_mag.shape} != phase shape {phases.shape}"
        
        num_frames = S_mag.shape[0]
        output_len = (num_frames - 1) * self.hop_len + self.frame_len
        
        output_signal = np.zeros(output_len, dtype=np.float32)
        window_norm_buffer = np.zeros(output_len, dtype=np.float32)
        
        WINDOW = self.synthesis_window  # Hann Window
        
        for t in range(num_frames):
            # Reconstruct complex FFT bins
            clean_spec = S_mag[t] * np.exp(1j * phases[t])
            
            # FORCE EXACT IFFT LENGTH: Explicitly pass self.frame_len (480) to prevent time stretching
            clean_time_frame = np.fft.irfft(clean_spec, n=self.frame_len)
            
            # Weighted Overlap-Add synthesis
            start_idx = t * self.hop_len
            end_idx = start_idx + self.frame_len
            
            # Multiply by synthesis window and accumulate
            output_signal[start_idx:end_idx] += clean_time_frame * WINDOW
            # Accumulate squared window weights to track overlap envelope
            window_norm_buffer[start_idx:end_idx] += WINDOW ** 2
            
        # Safely divide by norm buffer (handling zero overlap at boundary edges)
        window_norm_buffer[window_norm_buffer < 1e-9] = 1.0
        final_output = output_signal / window_norm_buffer
        
        # Clamp amplitude between [-1.0, 1.0] to prevent any clipping distortion
        final_output = np.clip(final_output, -1.0, 1.0)
        return final_output

    def save_wav_16bit(self, file_path, audio_data):
        """
        Saves float32 audio scaled back to 16-bit PCM WAV format.
        """
        pcm_audio = (audio_data * 32767.0).astype(np.int16)
        wavfile.write(file_path, self.sample_rate, pcm_audio)
