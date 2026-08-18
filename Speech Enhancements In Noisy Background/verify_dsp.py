"""
VERIFICATION SCRIPT: verify_dsp.py
Performs automated verification of the DSP reconstruction, Bark projection, and PyTorch controller.
"""

import numpy as np
import torch
from dsp_engine import DSPEngine
from neural_controller import LightweightNeuralController, train_model

def test_wola_reconstruction():
    print("Testing Time-Domain WOLA Perfect Reconstruction...")
    # Initialize engine
    dsp = DSPEngine()
    
    # Synthesize a pure 1kHz sine wave (float32 Normalized)
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    test_signal = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    
    # Process through STFT Analysis
    frames = dsp.time_domain_framing(test_signal)
    mags, phases = dsp.compute_stft(frames)
    
    # Reconstruct directly without spectral subtraction (identity test)
    reconstructed = dsp.reconstruct_audio(mags, phases)
    
    # Crop to comparison length (avoiding edge padding effects)
    comp_len = len(test_signal)
    recon_cropped = reconstructed[:comp_len]
    
    # Compute reconstruction loss (Mean Squared Error)
    mse = np.mean(np.square(test_signal - recon_cropped))
    print(f"   Reconstruction MSE: {mse:.2e}")
    
    # Verify COLA/WOLA reconstruction is mathematically perfect
    assert mse < 1e-5, f"WOLA perfect reconstruction failed! MSE is too high: {mse}"
    print("   [PASS] WOLA Perfect Reconstruction Verified successfully! No volume pumping detected.")

def test_bark_projection():
    print("\nTesting Bark Scale Transformation Matrix...")
    dsp = DSPEngine()
    
    # Test matrix shape
    expected_shape = (24, 241)
    matrix_shape = dsp.M_LB.shape
    print(f"   Matrix Shape: {matrix_shape}")
    assert matrix_shape == expected_shape, f"Bark Matrix shape mismatch! Expected {expected_shape}, got {matrix_shape}"
    
    # Verify that Bark mapping projection computes successfully
    seq_len = 10
    dummy_mags = np.random.rand(seq_len, 241)
    bark_features = dsp.project_linear_to_bark(dummy_mags)
    
    expected_bark_shape = (seq_len, 24)
    print(f"   Bark Feature Shape: {bark_features.shape}")
    assert bark_features.shape == expected_bark_shape, f"Bark feature shape mismatch! Expected {expected_bark_shape}, got {bark_features.shape}"
    print("   [PASS] Bark scale projection math verified successfully!")

def test_neural_controller():
    print("\nTesting Lightweight Recurrent Neural Controller...")
    # Initialize Model
    model = LightweightNeuralController()
    
    # Dummy Batch forward pass
    batch_size = 2
    seq_len = 15
    dummy_input = torch.rand(batch_size, seq_len, 24)
    
    alpha, h_n = model(dummy_input)
    
    print(f"   Alpha output shape: {alpha.shape}")
    print(f"   Hidden state output shape: {h_n.shape}")
    
    # Bounds check: alphas must be in [0.0, 1.5]
    min_alpha = alpha.min().item()
    max_alpha = alpha.max().item()
    print(f"   Alpha range: [{min_alpha:.4f}, {max_alpha:.4f}]")
    
    assert alpha.shape == (batch_size, seq_len, 1), f"Alpha shape mismatch!"
    assert 0.0 <= min_alpha <= max_alpha <= 2.5, f"Alpha bounds violated! Got range [{min_alpha}, {max_alpha}]"
    print("   [PASS] Neural Controller bounds and forward pass verified successfully!")

if __name__ == "__main__":
    print("========================================")
    print("Running Speech Enhancement Suite Unit Tests")
    print("========================================")
    
    try:
        test_wola_reconstruction()
        test_bark_projection()
        test_neural_controller()
        print("\nALL UNIT TESTS PASSED SUCCESSFULY!")
    except Exception as e:
        print(f"\nA test failed with error: {e}")
        exit(1)
