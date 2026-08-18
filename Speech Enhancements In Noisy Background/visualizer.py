"""
MODULE 4: METRICS & COMPARATIVE GRAPH VISUALIZATION (Member 4 Domain)
Implements physical metric computations and renders professional, dark-themed audio analysis plots.
"""

import numpy as np
import matplotlib.pyplot as plt

# Apply a professional dark theme for all matplotlib plots
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.facecolor': '#0F111A',       # Premium dark midnight blue
    'figure.facecolor': '#0F111A',     # Match stream lit dark background
    'grid.color': '#1F2937',           # Subtle gray gridlines
    'text.color': '#E5E7EB',
    'axes.labelcolor': '#9CA3AF',
    'xtick.color': '#9CA3AF',
    'ytick.color': '#9CA3AF',
    'axes.edgecolor': '#374151'
})

def calculate_noise_power_metrics(noisy_mags, enhanced_mags):
    """
    Calculates the average "Before Noise Power" and "After Noise Power" 
    by integrating the Power Spectral Density (PSD) of estimated silent segments 
    before and after subtraction.
    
    Silent segments are estimated as the 10% frames with the lowest total energy.
    """
    # Calculate energy per frame (sum of squared magnitudes)
    frame_energies = np.sum(np.square(noisy_mags), axis=1)
    
    # Identify indices of the 10% lowest energy frames (silence/noise floor baseline)
    num_frames = len(frame_energies)
    num_silent_frames = max(1, int(num_frames * 0.10))
    silent_indices = np.argsort(frame_energies)[:num_silent_frames]
    
    # Calculate Power Spectral Density (PSD) for these silent frames
    noisy_psd_silent = np.square(noisy_mags[silent_indices])
    enhanced_psd_silent = np.square(enhanced_mags[silent_indices])
    
    # Average PSD over the silent frames
    avg_noisy_psd = np.mean(noisy_psd_silent, axis=0)
    avg_enhanced_psd = np.mean(enhanced_psd_silent, axis=0)
    
    # Integrate PSD (sum over all frequency bins) to get total noise power
    before_power = np.sum(avg_noisy_psd)
    after_power = np.sum(avg_enhanced_psd)
    
    # Convert to decibel relative scale (to make it readable)
    before_db = 10 * np.log10(before_power + 1e-10)
    after_db = 10 * np.log10(after_power + 1e-10)
    
    return before_db, after_db, before_power, after_power

def plot_time_domain_waveforms(noisy_audio, enhanced_audio, sample_rate=16000):
    """
    Plot A: Time-Domain Waveforms
    Two stacked line plots comparing the clean signal envelope vs the noisy raw envelope over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, sharey=True)
    
    time_noisy = np.arange(len(noisy_audio)) / sample_rate
    time_enhanced = np.arange(len(enhanced_audio)) / sample_rate
    
    # Compute audio amplitude envelopes using simple rolling maximum
    window_size = int(0.02 * sample_rate) # 20ms window
    noisy_envelope = np.zeros_like(noisy_audio)
    enhanced_envelope = np.zeros_like(enhanced_audio)
    
    # Fast vectorized rolling max approximation
    for i in range(0, len(noisy_audio), window_size):
        end = min(i + window_size, len(noisy_audio))
        noisy_envelope[i:end] = np.max(np.abs(noisy_audio[i:end]))
        enhanced_envelope[i:end] = np.max(np.abs(enhanced_audio[i:end]))
        
    # Plot noisy waveform and envelope
    ax1.plot(time_noisy, noisy_audio, color='#E11D48', alpha=0.3, label='Noisy Waveform') # Crimson Rose
    ax1.plot(time_noisy, noisy_envelope, color='#F43F5E', alpha=0.9, linewidth=1.5, label='Signal Envelope')
    ax1.set_title('Noisy Raw Signal Envelope (Before Processing)', fontsize=11, color='#F43F5E', fontweight='semibold')
    ax1.set_ylabel('Amplitude', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', frameon=False, fontsize=8)
    
    # Plot enhanced waveform and envelope
    ax2.plot(time_enhanced, enhanced_audio, color='#059669', alpha=0.3, label='Enhanced Waveform') # Vibrant Emerald
    ax2.plot(time_enhanced, enhanced_envelope, color='#10B981', alpha=0.9, linewidth=1.5, label='Signal Envelope')
    ax2.set_title('Neural-Enhanced Clean Envelope (After Processing)', fontsize=11, color='#10B981', fontweight='semibold')
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_spectrograms(noisy_mags, enhanced_mags, sample_rate=16000, hop_len=240):
    """
    Plot B: Frequency-Domain Spectrograms
    Two heatmaps displaying frequency vs time to visually demonstrate the erasure of background noise bands.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    
    # Transpose magnitude spectra for standard spectrogram plotting (Frequency on Y, Time on X)
    noisy_db = 20 * np.log10(noisy_mags.T + 1e-5)
    enhanced_db = 20 * np.log10(enhanced_mags.T + 1e-5)
    
    num_frames = noisy_mags.shape[0]
    total_duration = (num_frames * hop_len) / sample_rate
    
    # Determine frequency bins limits (0 to 8kHz)
    extent = [0, total_duration, 0, sample_rate / 2000.0] # Y axis in kHz
    
    # Color map 'magma' or 'inferno' displays noise floor beautifully
    im1 = ax1.imshow(noisy_db, aspect='auto', origin='lower', cmap='magma', extent=extent, vmin=-60, vmax=10)
    ax1.set_title('Noisy Spectrogram (Before)', fontsize=11, color='#F43F5E', fontweight='semibold')
    ax1.set_xlabel('Time (seconds)', fontsize=9)
    ax1.set_ylabel('Frequency (kHz)', fontsize=9)
    ax1.grid(False)
    
    im2 = ax2.imshow(enhanced_db, aspect='auto', origin='lower', cmap='magma', extent=extent, vmin=-60, vmax=10)
    ax2.set_title('Clean Spectrogram (After)', fontsize=11, color='#10B981', fontweight='semibold')
    ax2.set_xlabel('Time (seconds)', fontsize=9)
    ax2.grid(False)
    
    # Add a unified vertical color bar on the right side
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Power Spectral Density (dB)', fontsize=8)
    
    return fig

def plot_psd_profiles(noisy_mags, enhanced_mags, sample_rate=16000):
    """
    Plot C: Average Power Spectral Density (PSD) Profile
    A line chart plotting frequency against power to verify that the noise floor dropped 
    across Bark bands while speech peaks remained intact.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Compute average PSD profile over the entire sequence duration
    avg_noisy_psd = np.mean(np.square(noisy_mags), axis=0)
    avg_enhanced_psd = np.mean(np.square(enhanced_mags), axis=0)
    
    # Convert to relative dB scale
    noisy_db = 10 * np.log10(avg_noisy_psd + 1e-10)
    enhanced_db = 10 * np.log10(avg_enhanced_psd + 1e-10)
    
    num_bins = len(avg_noisy_psd)
    freqs = np.linspace(0, sample_rate / 2.0, num_bins)
    
    ax.plot(freqs, noisy_db, color='#F43F5E', linewidth=1.5, label='Noisy Raw Signal')
    ax.plot(freqs, enhanced_db, color='#10B981', linewidth=1.8, label='Neural Enhanced Signal')
    
    # Add beautiful translucent fills underneath curves
    ax.fill_between(freqs, noisy_db, -100, color='#F43F5E', alpha=0.08)
    ax.fill_between(freqs, enhanced_db, -100, color='#10B981', alpha=0.05)
    
    ax.set_title('Average Power Spectral Density (PSD) Profile', fontsize=11, fontweight='semibold')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power (dB)', fontsize=10)
    ax.set_xlim(0, sample_rate / 2.0)
    ax.set_ylim(-80, 5)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.legend(loc='upper right', frameon=True, facecolor='#111827', edgecolor='#374151')
    
    return fig
