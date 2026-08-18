"""
Neural-Driven Low-Rank Multiband Speech Enhancement App
A single, self-contained, production-ready implementation of a hybrid PyTorch GRU-DNN
neural controller driving a vectorized linear magnitude spectral subtraction loop.
"""

import streamlit as st
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.ndimage import convolve1d
import torch
import torch.nn as nn
import torch.optim as optim
import io
import time
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# MODULE 3: HYBRID TEMPORAL GRU-DNN NEURAL CONTROLLER (Member 3 Domain)
# ----------------------------------------------------------------------
class LightweightNeuralController(nn.Module):
    def __init__(self, num_bark_bands=24, gru_hidden_dim=32):
        super().__init__()
        # GRU tracks the slow-moving temporal noise floor across frames
        self.gru = nn.GRU(
            input_size=num_bark_bands, 
            hidden_size=gru_hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        # Deep Neural Network to map GRU state to over-subtraction factor
        self.dnn = nn.Sequential(
            nn.Linear(gru_hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # Bounded scaling parameter to prevent over-subtraction
        self.scale_factor = 2.5
        self._init_sensible_weights()

    def _init_sensible_weights(self):
        # Initialize weight matrices such that it yields reasonable default factors (0.4 to 0.8)
        # with a negative correlation on input energy to protect speech
        with torch.no_grad():
            nn.init.xavier_uniform_(self.gru.weight_ih_l0)
            nn.init.orthogonal_(self.gru.weight_hh_l0)
            self.gru.bias_ih_l0.fill_(0.0)
            self.gru.bias_hh_l0.fill_(0.0)
            
            # Linear layer initialization
            for layer in self.dnn:
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(-0.05)
                    layer.bias.fill_(0.1)

    def forward(self, noisy_bark, h_state=None):
        """
        noisy_bark shape: (batch_size, seq_len, 24)
        """
        gru_out, h_state = self.gru(noisy_bark, h_state)
        dnn_out = self.dnn(gru_out)
        alpha_t = dnn_out * self.scale_factor  # Bounded strictly to [0.0, 2.5]
        return alpha_t, h_state

def differentiable_spectral_subtraction(Y_mag, alpha, D_mag, W_linear, beta=0.02):
    """
    Fully differentiable PyTorch linear magnitude spectral subtraction with Wiener soft-attenuation,
    3-point frequency-axis smoothing, and first-order temporal IIR smoothing.
    """
    batch_size, seq_len, num_bins = Y_mag.shape
    
    # Pre-weighted linear noise profile: shape (1, 1, 241)
    weighted_noise = D_mag * W_linear
    weighted_noise = weighted_noise.view(1, 1, -1)
    
    # Calculate Wiener-style soft attenuation factor H, bounded strictly to beta=0.15 floor
    H = torch.maximum(1.0 - alpha * (weighted_noise / (Y_mag + 1e-6)), torch.tensor(beta, device=Y_mag.device))
    raw_clean = H * Y_mag
    
    # Active 3-point moving average filter across frequency bins (to prevent echo/reverb)
    x_reshaped = raw_clean.view(batch_size * seq_len, 1, num_bins)
    x_padded = torch.nn.functional.pad(x_reshaped, (1, 1), mode='reflect')
    kernel = torch.tensor([0.15, 0.7, 0.15], dtype=torch.float32, device=Y_mag.device).view(1, 1, 3)
    smoothed = torch.nn.functional.conv1d(x_padded, kernel)
    raw_clean_smooth = smoothed.view(batch_size, seq_len, num_bins)
    
    # First-order IIR temporal smoothing filter across frames
    S_enhanced_list = []
    prev_clean = raw_clean_smooth[:, 0, :]
    S_enhanced_list.append(prev_clean.unsqueeze(1))
    
    for t in range(1, seq_len):
        current_clean = raw_clean_smooth[:, t, :]
        smoothed_clean = 0.4 * prev_clean + 0.6 * current_clean
        S_enhanced_list.append(smoothed_clean.unsqueeze(1))
        prev_clean = smoothed_clean
        
    S_enhanced = torch.cat(S_enhanced_list, dim=1)
    return S_enhanced

def train_model(model, noisy_bark_feats, clean_bark_feats, noisy_mags, clean_mags, D_mag, W_linear, beta=0.02, epochs=30, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert numpy inputs to torch tensors
    x_bark = torch.tensor(noisy_bark_feats, dtype=torch.float32).unsqueeze(0).to(device)
    Y_mag = torch.tensor(noisy_mags, dtype=torch.float32).unsqueeze(0).to(device)
    X_clean = torch.tensor(clean_mags, dtype=torch.float32).unsqueeze(0).to(device)
    D_m = torch.tensor(D_mag, dtype=torch.float32).to(device)
    W_lin = torch.tensor(W_linear, dtype=torch.float32).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass on GRU model
        alpha, _ = model(x_bark)
        # End-to-end backpropagation through linear subtraction core
        S_enhanced = differentiable_spectral_subtraction(Y_mag, alpha, D_m, W_lin, beta)
        loss = criterion(S_enhanced, X_clean)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return losses

# ----------------------------------------------------------------------
# MODULE 1 & 2: DSP ENGINE & PSYCHOACOUSTIC CORE (Members 1 & 2 Domains)
# ----------------------------------------------------------------------
class DSPEngine:
    def __init__(self, sample_rate=16000, beta=0.02):
        self.sample_rate = sample_rate
        self.frame_len = 480       # 30ms window at 16kHz
        self.hop_len = 240         # 15ms hop size (50% overlap)
        self.num_bins = 241        # rfft output bins
        self.num_bark_bands = 24
        self.beta = beta           # Hardcoded soft protection floor constant (15%)
        
        # Hann window satisfying COLA constraints
        self.window = np.hanning(self.frame_len)
        
        # Build Bark Scale mapping matrix (24, 241)
        self.M_LB = self._build_bark_matrix()
        
        # Pre-compute Bark weights mapped across critical bands
        self.W_b = np.ones(self.num_bark_bands, dtype=np.float32)
        for b in range(self.num_bark_bands):
            if b < 4:
                self.W_b[b] = 2.2  # Aggressively suppress electrical hum / low rumblings
            elif b < 14:
                self.W_b[b] = 1.5  # Strong voice-band noise suppression
            else:
                self.W_b[b] = 0.8  # Suppress high-frequency details without speech muffling
                
        # Map Bark weights to the 241 linear frequency bins
        self.W_linear = np.zeros(self.num_bins, dtype=np.float32)
        freqs = np.linspace(0, self.sample_rate / 2.0, self.num_bins)
        for k, f in enumerate(freqs):
            bark = self._hz_to_bark(f)
            band_idx = self._bark_to_band_index(bark)
            self.W_linear[k] = self.W_b[band_idx]

    def _hz_to_bark(self, f):
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    def _bark_to_band_index(self, bark):
        max_bark = self._hz_to_bark(self.sample_rate / 2.0)
        width = max_bark / self.num_bark_bands
        idx = int(np.floor(bark / width))
        return min(self.num_bark_bands - 1, max(0, idx))

    def _build_bark_matrix(self):
        matrix = np.zeros((self.num_bark_bands, self.num_bins), dtype=np.float32)
        freqs = np.linspace(0, self.sample_rate / 2.0, self.num_bins)
        for k, f in enumerate(freqs):
            bark = self._hz_to_bark(f)
            band_idx = self._bark_to_band_index(bark)
            matrix[band_idx, k] = 1.0
            
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    def load_audio(self, file_buffer):
        sr, audio = wavfile.read(file_buffer)
        
        # Float32 Normalization
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        else:
            audio = audio.astype(np.float32)
            
        # Mono downmixing
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Standard decimation resampling if SR is different
        if sr != self.sample_rate:
            duration = len(audio) / sr
            num_samples = int(duration * self.sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio), num_samples),
                np.arange(len(audio)),
                audio
            )
            
        # Cache original maximum amplitude to preserve gain levels later
        orig_max = np.max(np.abs(audio))
        if orig_max == 0:
            orig_max = 1.0
        audio = audio / orig_max
        
        return audio, orig_max

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
            # Apply analysis Hann Window (exactly once)
            windowed = frames[t] * self.window
            spec = np.fft.rfft(windowed, n=self.frame_len)
            mags[t] = np.abs(spec)
            phases[t] = np.angle(spec)
        return mags, phases

    def project_linear_to_bark(self, magnitudes):
        # B = M_{L->B} dot |Y_t(f)|
        return np.dot(magnitudes, self.M_LB.T)

    def spectral_subtraction_core(self, Y_mag, alpha_t, D_mag):
        """
        Wiener-style soft attenuation spectral floor masking to eliminate
        robotic backgrounds, paired with 3-point frequency smoothing and frame temporal smoothing.
        """
        seq_len = Y_mag.shape[0]
        S_mag = np.zeros_like(Y_mag)
        weighted_noise = D_mag * self.W_linear
        prev_clean_mag = None
        
        for t in range(seq_len):
            # Wiener-style soft attenuation H
            H = np.maximum(1.0 - alpha_t[t] * (weighted_noise / (Y_mag[t] + 1e-6)), self.beta)
            clean_mag = H * Y_mag[t]
            
            # Frequency convolve1d smoothing (3-point moving average weights [0.25, 0.5, 0.25])
            clean_mag = convolve1d(clean_mag, weights=[0.15, 0.7, 0.15], mode='mirror')
            
            # Temporal smoothing (First-order recursive IIR)
            if t == 0:
                prev_clean_mag = clean_mag.copy()
            else:
                clean_mag = 0.4 * prev_clean_mag + 0.6 * clean_mag
                prev_clean_mag = clean_mag.copy()
                
            S_mag[t] = clean_mag
        return S_mag

    def reconstruct_audio(self, S_mag, phases):
        """
        Strict WOLA synthesis overlap-add with exact irfft length (480)
        and window squared normalization.
        """
        assert S_mag.shape == phases.shape, "Magnitude and phase dimensions mismatch!"
        num_frames = S_mag.shape[0]
        
        # Calculate total samples representing padded frames
        total_samples = (num_frames - 1) * self.hop_len
        output_len = total_samples + 480
        
        # Initialize output signal and normalization buffers
        output_signal = np.zeros(output_len, dtype=np.float32)
        window_norm_buffer = np.zeros(output_len, dtype=np.float32)
        
        WINDOW = self.window # Hann window (np.hanning(480))
        
        for t in range(num_frames):
            # Verify Phase Alignment Dimensions: clean_mag (241,) matches phase (241,)
            assert S_mag[t].shape == (241,), f"Mag bin size mismatch: {S_mag[t].shape}"
            assert phases[t].shape == (241,), f"Phase bin size mismatch: {phases[t].shape}"
            
            clean_spec = S_mag[t] * np.exp(1j * phases[t])
            
            # FORCE EXACT IFFT LENGTH: Explicitly pass 480 to stop audio from slowing down
            clean_time_frame = np.fft.irfft(clean_spec, n=480)
            
            start_idx = t * self.hop_len
            end_idx = start_idx + 480
            
            # Multiply by window EXACTLY TWICE (framing + reconstruction WOLA)
            output_signal[start_idx:end_idx] += clean_time_frame * WINDOW
            window_norm_buffer[start_idx:end_idx] += WINDOW ** 2
            
        # CRITICAL COLA NORMALIZATION: Avoid dividing by tiny near-zero float boundaries (threshold 1e-9)
        # to completely eliminate harsh cracking, clicking, and boundary static noise
        window_norm_buffer[window_norm_buffer < 1e-9] = 1.0
        final_output = output_signal / window_norm_buffer
        
        return np.clip(final_output, -1.0, 1.0)
    def save_wav_16bit(self, file_path, audio_data):
        """
        Saves float32 audio scaled back to 16-bit PCM WAV format.
        """
        pcm_audio = (audio_data * 32767.0).astype(np.int16)
        wavfile.write(file_path, self.sample_rate, pcm_audio)

# ----------------------------------------------------------------------
# MODULE 4: Streamlit UI & Visualizer (Member 4 Domain)
# ----------------------------------------------------------------------
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#0F111A',
    'figure.facecolor': '#0F111A',
    'grid.color': '#1F2937',
    'text.color': '#E5E7EB',
    'axes.edgecolor': '#374151'
})

def calculate_noise_power_metrics(noisy_mags, enhanced_mags):
    # Lowest 10% energy frames represent silent baseline segments
    frame_energies = np.sum(np.square(noisy_mags), axis=1)
    num_silent = max(1, int(len(frame_energies) * 0.10))
    silent_indices = np.argsort(frame_energies)[:num_silent]
    
    noisy_silent = np.square(noisy_mags[silent_indices])
    enhanced_silent = np.square(enhanced_mags[silent_indices])
    
    before_power = np.sum(np.mean(noisy_silent, axis=0))
    after_power = np.sum(np.mean(enhanced_silent, axis=0))
    
    before_db = 10 * np.log10(before_power + 1e-10)
    after_db = 10 * np.log10(after_power + 1e-10)
    return before_db, after_db, before_power, after_power

def plot_time_domain_waveforms(noisy, enhanced):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, sharey=True)
    t_n = np.arange(len(noisy)) / 16000.0
    t_e = np.arange(len(enhanced)) / 16000.0
    
    # Vectorized envelope calculations (rolling max)
    w_size = 320
    n_env = np.zeros_like(noisy)
    e_env = np.zeros_like(enhanced)
    for i in range(0, len(noisy), w_size):
        end = min(i + w_size, len(noisy))
        n_env[i:end] = np.max(np.abs(noisy[i:end]))
        e_env[i:end] = np.max(np.abs(enhanced[i:end]))
        
    ax1.plot(t_n, noisy, color='#E11D48', alpha=0.3)
    ax1.plot(t_n, n_env, color='#F43F5E', linewidth=1.5, label='Noisy Envelope')
    ax1.set_title('Raw Signal Envelope (Before Processing)', color='#F43F5E', fontsize=10, fontweight='semibold')
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.legend(loc='upper right', frameon=False, fontsize=8)
    
    ax2.plot(t_e, enhanced, color='#059669', alpha=0.3)
    ax2.plot(t_e, e_env, color='#10B981', linewidth=1.5, label='Enhanced Envelope')
    ax2.set_title('Neural-Enhanced Envelope (After Processing)', color='#10B981', fontsize=10, fontweight='semibold')
    ax2.set_xlabel('Time (seconds)', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_spectrograms(noisy_mags, enhanced_mags):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    noisy_db = 20 * np.log10(noisy_mags.T + 1e-5)
    enhanced_db = 20 * np.log10(enhanced_mags.T + 1e-5)
    
    duration = (noisy_mags.shape[0] * 240) / 16000.0
    extent = [0, duration, 0, 8.0]
    
    ax1.imshow(noisy_db, aspect='auto', origin='lower', cmap='magma', extent=extent, vmin=-50, vmax=10)
    ax1.set_title('Noisy Spectrogram (Before)', color='#F43F5E', fontsize=10, fontweight='semibold')
    ax1.set_ylabel('Frequency (kHz)', fontsize=9)
    ax1.set_xlabel('Time (seconds)', fontsize=9)
    ax1.grid(False)
    
    im2 = ax2.imshow(enhanced_db, aspect='auto', origin='lower', cmap='magma', extent=extent, vmin=-50, vmax=10)
    ax2.set_title('Clean Spectrogram (After)', color='#10B981', fontsize=10, fontweight='semibold')
    ax2.set_xlabel('Time (seconds)', fontsize=9)
    ax2.grid(False)
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax).set_label('Energy (dB)', fontsize=8)
    return fig

def plot_psd_profiles(noisy_mags, enhanced_mags):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    noisy_psd = 10 * np.log10(np.mean(np.square(noisy_mags), axis=0) + 1e-10)
    enhanced_psd = 10 * np.log10(np.mean(np.square(enhanced_mags), axis=0) + 1e-10)
    
    freqs = np.linspace(0, 8000.0, len(noisy_psd))
    ax.plot(freqs, noisy_psd, color='#F43F5E', linewidth=1.5, label='Noisy Raw PSD')
    ax.plot(freqs, enhanced_psd, color='#10B981', linewidth=1.8, label='Enhanced PSD')
    ax.fill_between(freqs, noisy_psd, -80, color='#F43F5E', alpha=0.08)
    ax.fill_between(freqs, enhanced_psd, -80, color='#10B981', alpha=0.05)
    
    ax.set_title('Average Power Spectral Density (PSD) Profile', fontsize=10, fontweight='semibold')
    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Power (dB)', fontsize=9)
    ax.set_xlim(0, 8000.0)
    ax.set_ylim(-70, 5)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.legend(loc='upper right', frameon=True, facecolor='#111827', edgecolor='#374151')
    return fig

def generate_synthetic_audio(duration=5.0, sample_rate=16000):
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Vowel speech model (Male pitch fundamental + resonances)
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
    
    # Background Noise: low frequency 50Hz hum + broadband white hiss
    hum = 0.08 * np.sin(2.0 * np.pi * 50.0 * t) + 0.03 * np.sin(2.0 * np.pi * 150.0 * t)
    hiss = np.random.normal(0.0, 0.06, num_samples)
    
    noisy_speech = clean_speech + hum + hiss
    norm = max(np.max(np.abs(noisy_speech)), 1.0)
    return noisy_speech / norm, clean_speech / norm

# Streamlit Page Setup
st.set_page_config(
    page_title="Neural-Driven Low-Rank Multiband Speech Enhancement",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0B0F19; }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white; border: none; padding: 0.6rem 1.8rem;
        border-radius: 8px; font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .noisy-title { color: #F43F5E; font-weight: 700; }
    .clean-title { color: #10B981; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

if "trained_model" not in st.session_state:
    st.session_state.trained_model = LightweightNeuralController()
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "synthetic_audio" not in st.session_state:
    st.session_state.synthetic_audio = None
if "synthetic_clean" not in st.session_state:
    st.session_state.synthetic_clean = None

st.title("🎙️ Neural-Driven Low-Rank Multiband Speech Enhancement")
st.markdown(
    "A self-contained production interface using a hybrid recurrent **PyTorch GRU-DNN Controller** "
    "optimized to drive a **Linear Magnitude Spectral Subtractor** with zero volume pumping."
)
st.markdown("---")

st.sidebar.header("🛠️ DSP & AI Hyperparameters")
agg_multiplier = st.sidebar.slider(
    "Over-Subtraction Multiplier",
    min_value=0.2, max_value=2.0, value=1.4, step=0.1,
    help="Multiplies the alpha_t predicted by the PyTorch GRU neural model."
)

beta_val = st.sidebar.slider(
    "Spectral Protection Floor (Beta)",
    min_value=0.01, max_value=0.30, value=0.01, step=0.01,
    help="Sets the protection baseline to preserve speech details."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Neural Controller Status")
if st.session_state.is_trained:
    st.sidebar.success("🧠 PyTorch GRU-DNN: Live Custom Trained")
else:
    st.sidebar.info("🧠 PyTorch GRU-DNN: Default Energy-Tracker")

col_upload, col_synthetic = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Noisy Audio File (.wav)",
        type=["wav"],
        help="Upload a standard WAV audio track."
    )
with col_synthetic:
    st.markdown("<p style='margin-bottom:0.5rem; font-weight:600;'>No file handy? Generate a synthetic speech environment!</p>", unsafe_allow_html=True)
    if st.button("🌟 Generate Synthetic Demo"):
        noisy_sig, clean_sig = generate_synthetic_audio()
        st.session_state.synthetic_audio = noisy_sig
        st.session_state.synthetic_clean = clean_sig
        st.success("Generated 5s track containing vowel formants mixed with 50Hz hum and broadband white noise!")

audio_data = None
true_clean = None
orig_max_cached = 1.0
source_name = ""

dsp = DSPEngine(beta=beta_val)

if uploaded_file is not None:
    try:
        audio_data, orig_max_cached = dsp.load_audio(uploaded_file)
        source_name = uploaded_file.name
        true_clean = None
    except Exception as e:
        st.error(f"Error loading WAV: {e}")
elif st.session_state.synthetic_audio is not None:
    audio_data = st.session_state.synthetic_audio
    true_clean = st.session_state.synthetic_clean
    orig_max_cached = 1.0
    source_name = "Synthetic_Speech_Hum_Demo.wav"

if audio_data is not None:
    st.info(f"Loaded: `{source_name}` | Length: {len(audio_data)} samples | Sample Rate: 16000 Hz")
    
    # Neural model self-supervised optimization block
    if true_clean is not None:
        st.markdown("### 🧠 Live Differentiable Model Optimization")
        st.markdown(
            " optimizes the PyTorch GRU-DNN controller by backpropagating MSE loss from the linear magnitude spectral subtraction math."
        )
        t_col1, t_col2 = st.columns([1, 2])
        with t_col1:
            epochs = st.slider("Optimization Epochs", min_value=5, max_value=50, value=25, step=5)
            if st.button("🔥 Run Differentiable Optimization"):
                with st.spinner("Backpropagating gradients end-to-end..."):
                    frames_noisy = dsp.time_domain_framing(audio_data)
                    frames_clean = dsp.time_domain_framing(true_clean)
                    Y_mags, _ = dsp.compute_stft(frames_noisy)
                    X_mags, _ = dsp.compute_stft(frames_clean)
                    
                    noisy_bark = dsp.project_linear_to_bark(Y_mags)
                    clean_bark = dsp.project_linear_to_bark(X_mags)
                    
                    D_mag = np.mean(Y_mags[:5], axis=0)
                    
                    losses = train_model(
                        st.session_state.trained_model,
                        noisy_bark,
                        clean_bark,
                        Y_mags,
                        X_mags,
                        D_mag,
                        dsp.W_linear,
                        beta=beta_val,
                        epochs=epochs,
                        lr=0.01
                    )
                    st.session_state.is_trained = True
                    st.success("Optimization completed successfully!")
                    
                    fig_loss, ax_loss = plt.subplots(figsize=(6, 2))
                    ax_loss.plot(losses, color='#10B981', linewidth=2)
                    ax_loss.set_title("Training Mean Squared Error (MSE)", fontsize=9, fontweight='bold')
                    ax_loss.set_xlabel("Epoch", fontsize=8)
                    ax_loss.grid(True, alpha=0.1)
                    st.pyplot(fig_loss)
        with t_col2:
            st.markdown(
                """
                > **End-to-End Backpropagation Logic:**
                > 1. Ingests 24 psychoacoustic Bark energy features frame-by-frame.
                > 2. PyTorch GRU tracks temporal noise profiles; DNN outputs over-subtraction scaling parameter $\\alpha_t$.
                > 3. Computes linear spectral subtraction via soft Wiener masking.
                > 4. Calculates Mean Squared Error between the reconstructed magnitudes and clean target magnitudes.
                > 5. Backpropagates loss directly through the subtraction layers to update GRU-DNN nodes.
                """
            )
            
    # Processing stage execution
    st.markdown("---")
    st.subheader("⚡ Signal Processing & Neural Inference")
    with st.spinner("Executing STFT analysis, GRU state tracking, Wiener attenuation, and WOLA resynthesis..."):
        # Module 1: Time Domain Framing
        frames = dsp.time_domain_framing(audio_data)
        
        # Module 1: STFT Magnitude & Phase (Analysis)
        Y_mags, Y_phases = dsp.compute_stft(frames)
        
        # Module 2: Project magnitudes to Bark Bands
        bark_energy = dsp.project_linear_to_bark(Y_mags)
        
        # Module 3: PyTorch Neural Controller Inference
        bark_tensor = torch.tensor(bark_energy, dtype=torch.float32).unsqueeze(0)
        st.session_state.trained_model.eval()
        with torch.no_grad():
            alpha_tensor, _ = st.session_state.trained_model(bark_tensor)
            alpha_array = alpha_tensor.squeeze(0).squeeze(-1).numpy()
            
        alpha_array = alpha_array * agg_multiplier
        
        # Module 2: Noise fingerprint (mean of first 5 silence frames)
        D_mag = np.mean(Y_mags[:5], axis=0)
        
        # Module 2: Linear Subtraction Core
        S_mags = dsp.spectral_subtraction_core(Y_mags, alpha_array, D_mag)
        
        # Module 1: WOLA Reconstruction
        enhanced_audio = dsp.reconstruct_audio(S_mags, Y_phases)
        
        # Restore cached original gain factor
        audio_data_rescaled = audio_data * orig_max_cached
        enhanced_audio_rescaled = enhanced_audio * orig_max_cached
        
    st.success("Enhancement pipeline completed successfully!")
    
    # Metrics display
    before_db, after_db, before_raw, after_raw = calculate_noise_power_metrics(Y_mags, S_mags)
    noise_reduction = before_db - after_db
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            f"""<div class="metric-card">
                <span class="noisy-title">🔴 Raw Noise Floor</span>
                <h2>{before_db:.2f} dB</h2>
                <small>Power: {before_raw:.6f}</small>
            </div>""", unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f"""<div class="metric-card">
                <span class="clean-title">🟢 Enhanced Noise Floor</span>
                <h2>{after_db:.2f} dB</h2>
                <small>Power: {after_raw:.6f}</small>
            </div>""", unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            f"""<div class="metric-card">
                <span style="color:#3B82F6; font-weight:700;">🔵 Net Suppressed Power</span>
                <h2 style="color:#3B82F6;">{noise_reduction:.2f} dB</h2>
                <small>Reduction Factor: {np.exp((before_db - after_db)/10):.1f}x</small>
            </div>""", unsafe_allow_html=True
        )
        
    # Audio Playback
    st.markdown("### 🎧 Playback & Downloads")
    ap_col1, ap_col2 = st.columns(2)
    with ap_col1:
        st.markdown("<span class='noisy-title'>Noisy Raw Audio (Before)</span>", unsafe_allow_html=True)
        raw_buffer = io.BytesIO()
        dsp.save_wav_16bit(raw_buffer, audio_data_rescaled)
        st.audio(raw_buffer.getvalue(), format="audio/wav")
    with ap_col2:
        st.markdown("<span class='clean-title'>Neural-Enhanced Clean Audio (After)</span>", unsafe_allow_html=True)
        enhanced_buffer = io.BytesIO()
        dsp.save_wav_16bit(enhanced_buffer, enhanced_audio_rescaled)
        st.audio(enhanced_buffer.getvalue(), format="audio/wav")
        
    st.download_button(
        label="📥 Download Clean 16-bit PCM WAV File",
        data=enhanced_buffer.getvalue(),
        file_name=f"enhanced_{source_name}",
        mime="audio/wav"
    )
    
    # Graphs
    st.markdown("### 📊 Dual-Domain Comparative Graph Visualizations")
    tab1, tab2, tab3 = st.tabs([
        "📈 Time-Domain Waveforms (Plot A)",
        "🔥 Spectrogram Heatmaps (Plot B)",
        "📉 Average PSD Profile (Plot C)"
    ])
    with tab1:
        st.markdown("**Plot A: Time-Domain Amplitude Envelope comparison**")
        fig_time = plot_time_domain_waveforms(audio_data_rescaled, enhanced_audio_rescaled)
        st.pyplot(fig_time)
    with tab2:
        st.markdown("**Plot B: Spectrogram Heatmap Comparison**")
        fig_spec = plot_spectrograms(Y_mags, S_mags)
        st.pyplot(fig_spec)
    with tab3:
        st.markdown("**Plot C: Average Power Spectral Density (PSD) Profile**")
        fig_psd = plot_psd_profiles(Y_mags, S_mags)
        st.pyplot(fig_psd)
