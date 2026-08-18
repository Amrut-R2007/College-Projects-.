"""
Test and Refine script: test_and_refine.py
Runs empirical tests on the speech enhancement pipeline to sweep for stronger noise suppression,
evaluating noise reduction and speech distortion across beta, alpha multipliers, and Bark weights.
"""

import numpy as np
from scipy.ndimage import convolve1d

class TestDSPEngine:
    def __init__(self, sample_rate=16000, beta=0.02, temp_alpha=0.4, freq_weights=[0.15, 0.7, 0.15], aggressive_weights=False):
        self.sample_rate = sample_rate
        self.frame_len = 480
        self.hop_len = 240
        self.num_bins = 241
        self.beta = beta
        self.temp_alpha = temp_alpha      # 0.4
        self.freq_weights = freq_weights  # [0.15, 0.7, 0.15]
        self.window = np.hanning(self.frame_len)
        
        self.num_bark_bands = 24
        self.W_b = np.ones(self.num_bark_bands, dtype=np.float32)
        for b in range(self.num_bark_bands):
            if aggressive_weights:
                if b < 4:
                    self.W_b[b] = 2.5  # Heavy de-hum
                elif b < 14:
                    self.W_b[b] = 1.6  # Heavy speech band noise suppression
                else:
                    self.W_b[b] = 1.0  # Mute high frequency hiss
            else:
                if b < 4:
                    self.W_b[b] = 1.6
                elif b < 14:
                    self.W_b[b] = 1.0
                else:
                    self.W_b[b] = 0.5

        # Map to linear
        self.W_linear = np.zeros(self.num_bins, dtype=np.float32)
        freqs = np.linspace(0, self.sample_rate / 2.0, self.num_bins)
        for k, f in enumerate(freqs):
            bark = 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
            width = (13.0 * np.arctan(0.00076 * 8000.0) + 3.5 * np.arctan((8000.0 / 7500.0) ** 2)) / 24
            idx = min(23, max(0, int(np.floor(bark / width))))
            self.W_linear[k] = self.W_b[idx]

    def time_domain_framing(self, audio):
        pad_len = self.frame_len - (len(audio) - self.frame_len) % self.hop_len
        padded = np.pad(audio, (0, pad_len), mode='constant')
        num_frames = (len(padded) - self.frame_len) // self.hop_len + 1
        frames = np.zeros((num_frames, self.frame_len), dtype=np.float32)
        for t in range(num_frames):
            start = t * self.hop_len
            frames[t] = padded[start:start + self.frame_len]
        return frames

    def compute_stft(self, frames):
        num_frames = frames.shape[0]
        mags = np.zeros((num_frames, self.num_bins), dtype=np.float32)
        phases = np.zeros((num_frames, self.num_bins), dtype=np.float32)
        for t in range(num_frames):
            windowed = frames[t] * self.window
            spec = np.fft.rfft(windowed, n=self.frame_len)
            mags[t] = np.abs(spec)
            phases[t] = np.angle(spec)
        return mags, phases

    def spectral_subtraction_core(self, Y_mag, alpha_t, D_mag):
        seq_len = Y_mag.shape[0]
        S_mag = np.zeros_like(Y_mag)
        weighted_noise = D_mag * self.W_linear
        prev_clean_mag = None
        
        for t in range(seq_len):
            H = np.maximum(1.0 - alpha_t[t] * (weighted_noise / (Y_mag[t] + 1e-6)), self.beta)
            clean_mag = H * Y_mag[t]
            clean_mag = convolve1d(clean_mag, weights=self.freq_weights, mode='mirror')
            
            if t == 0:
                prev_clean_mag = clean_mag.copy()
            else:
                clean_mag = self.temp_alpha * prev_clean_mag + (1.0 - self.temp_alpha) * clean_mag
                prev_clean_mag = clean_mag.copy()
                
            S_mag[t] = clean_mag
        return S_mag

def generate_synthetic_audio(duration=5.0, sample_rate=16000):
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # 1. Speech Synthesis
    f0 = 130.0
    speech = np.zeros(num_samples)
    for h in range(1, 15):
        amp = 1.0 / (h ** 1.2)
        f = f0 * h
        if abs(f - 500) < 150: amp *= 3.0
        elif abs(f - 1500) < 250: amp *= 2.0
        elif abs(f - 2500) < 300: amp *= 1.5
        speech += amp * np.sin(2.0 * np.pi * f * t)
        
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.45 * t))
    gate = (np.sin(2.0 * np.pi * 0.65 * t) > -0.2).astype(np.float32)
    clean_speech = speech * envelope * gate
    clean_speech = 0.4 * (clean_speech / (np.max(np.abs(clean_speech)) + 1e-8))
    
    # 2. Hum + Hiss Noise
    hum = 0.08 * np.sin(2.0 * np.pi * 50.0 * t) + 0.03 * np.sin(2.0 * np.pi * 150.0 * t)
    hiss = np.random.normal(0.0, 0.06, num_samples)
    
    noisy_speech = clean_speech + hum + hiss
    norm = max(np.max(np.abs(noisy_speech)), 1.0)
    return noisy_speech / norm, clean_speech / norm

def evaluate_parameters(beta, agg_mult, aggressive_weights):
    noisy_sig, clean_sig = generate_synthetic_audio()
    
    dsp = TestDSPEngine(beta=beta, aggressive_weights=aggressive_weights)
    
    frames_noisy = dsp.time_domain_framing(noisy_sig)
    frames_clean = dsp.time_domain_framing(clean_sig)
    
    Y_mags, Y_phases = dsp.compute_stft(frames_noisy)
    X_mags, _ = dsp.compute_stft(frames_clean)
    
    frame_energies = np.sum(Y_mags, axis=1)
    norm_energy = frame_energies / (np.max(frame_energies) + 1e-6)
    alpha_t = 2.0 * (1.0 - norm_energy) * agg_mult
    
    D_mag = np.mean(Y_mags[:5], axis=0)
    
    S_mags = dsp.spectral_subtraction_core(Y_mags, alpha_t, D_mag)
    
    # Noise reduction on silent segment
    before_noise = np.sum(np.mean(np.square(Y_mags[:5]), axis=0))
    after_noise = np.sum(np.mean(np.square(S_mags[:5]), axis=0))
    noise_reduction_db = 10 * np.log10(before_noise / (after_noise + 1e-10))
    
    # Speech distortion during active segment
    speech_indices = np.where(norm_energy > 0.4)[0]
    speech_distortion = np.mean(np.square(X_mags[speech_indices] - S_mags[speech_indices]))
    
    print(f"Beta={beta} | Agg-Mult={agg_mult} | Bark-Weights-Aggressive={aggressive_weights}")
    print(f"   Noise Floor Reduction: {noise_reduction_db:.2f} dB")
    print(f"   Speech Distortion MSE: {speech_distortion:.6f}")
    
    return noise_reduction_db, speech_distortion

if __name__ == "__main__":
    print("Running Parameter Sweep for Aggressive Noise Suppression...")
    print("----------------------------------------------------------")
    configs = [
        (0.02, 1.0, False),  # Baseline
        (0.01, 1.0, False),  # Lower floor
        (0.005, 1.0, False), # Very low floor
        (0.01, 1.5, False),  # Lower floor + oversub multiplier 1.5
        (0.005, 1.5, False), # Very low floor + oversub multiplier 1.5
        (0.01, 1.5, True),   # Lower floor + oversub 1.5 + aggressive bark weights
        (0.005, 1.8, True),  # Very low floor + oversub 1.8 + aggressive bark weights
    ]
    for b, am, aw in configs:
        evaluate_parameters(b, am, aw)
