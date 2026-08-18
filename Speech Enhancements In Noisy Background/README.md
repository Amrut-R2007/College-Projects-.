# Neural-Driven Multiband Speech Enhancement App

A production-ready Python application that combines **Deep Learning (PyTorch)** and **Digital Signal Processing (NumPy/SciPy)** to achieve real-time speech enhancement using **Neural-Driven Low-Rank Multiband Spectral Subtraction**.

This application is built from mathematical first-principles without using bloated high-level audio libraries like `torchaudio` or `librosa`. All framing, FFT, Bark projections, and reconstruction algorithms are custom-built for absolute speed and processing transparency.

---

## 🛠️ Architecture & Module Structure

The application is structured into four highly cohesive and decoupled modules:

### 1. `dsp_engine.py` (Modules 1 & 2)
Handles standard audio I/O, Short-Time Fourier Transform (STFT), Inverse Short-Time Fourier Transform (ISTFT) via a Weighted Overlap-Add (WOLA) synthesis buffer, Bark mapping, and the core subtraction loop.
*   **Time-Domain Framing**: Splits the signal into 30ms frames (480 samples @ 16kHz) with a 15ms hop size (240 samples, 50% overlap).
*   **Hann Windowing**: Applies analysis and synthesis Hann windows satisfying the Constant Overlap-Add (COLA) constraint.
*   **Fast FFT**: Computes `np.fft.rfft` to extract 241 linear frequency bins, separating into noisy magnitude spectrum $|Y_t(f)|$ and noisy phase spectrum $\angle Y_t(f)$.
*   **Bark Matrix Mapping**: Projects 241 linear bins to 24 critical Bark scale bands using a static matrix $M_{L->B}$ of shape (24, 241) via matrix dot product: $B = M_{L->B} \times |Y_t(f)|$.
*   **Linear Subtraction via Soft Attenuation**: Computes a Wiener-style soft attenuation factor $H$ for each frequency bin, ensuring it never drops below a soft background floor of 0.01:
    $$H = \max\left(1.0 - \alpha_t \cdot \frac{W_{\text{linear}} \cdot |D(f)|}{|Y_t(f)| + 10^{-6}}, 0.01\right)$$
    $$|S_t(f)| = H \cdot |Y_t(f)|$$
    where $|D(f)|$ is the baseline static noise fingerprint magnitude calculated from the first 5 frames, and $W_{\text{linear}}$ is the Bark-scale weight vector mapped to the 241 linear bins.
*   **Magnitude Smoothing (Anti-Echo)**: Applies an active 3-point moving average filter across the frequency axis of the clean magnitude spectrum to round off sharp spectral edges and prevent phase tearing echo:
    $$|S_t(f)|_{\text{smooth}} = \text{Convolve}\left(|S_t(f)|, [0.15, 0.7, 0.15]\right)$$
*   **Temporal Smoothing**: Implements a first-order recursive IIR temporal smoothing filter across frames to prevent sudden spectral energy changes and eliminate musical noise:
    $$|S_t(f)|_{\text{smoothed}} = 0.4 \cdot |S_{t-1}(f)|_{\text{smoothed}} + 0.6 \cdot |S_t(f)|_{\text{smooth}}$$
*   **WOLA Reconstruction**: Uses `np.fft.irfft` to return to the time domain, accumulating windowed overlapping frames into a synthesis buffer and dividing strictly by the sum of overlapping squared Hann window weights to perfectly satisfy the COLA constraint and eliminate frame boundary amplitude modulations. The windowed frame is multiplied by the Hann window exactly twice (once at analysis STFT framing and once at synthesis reconstruction WOLA) for seamless transitions. To prevent severe boundary cracking or popping noise, normalization sums smaller than $10^{-9}$ are set to $1.0$ before division.

### 2. `neural_controller.py` (Module 3)
Defines the recurrent neural model and the differentiable end-to-end backpropagation pipeline.
*   **Model Structure**: A lightweight `nn.GRU` (24 inputs, 32 hidden states, 1 layer) followed by an `nn.Linear` fully-connected layer and a scaled `nn.Sigmoid` activation bounding the frame over-subtraction factor $\alpha_t$ strictly between $[0.0, 2.5]$.
*   **Differentiable DSP Subtraction**: Implements the spectral subtraction formula using PyTorch tensor operations, allowing gradients to propagate from the audio domain back through the traditional math equations into the neural network weights.

### 3. `visualizer.py` (Module 4)
Handles physical metric calculations and plots.
*   **Noise Power Metrics**: Calculates the relative decibel (dB) noise floor before and after subtraction by isolating the bottom 10% lowest energy frames (silence/noise profiles) and integrating their Power Spectral Density (PSD).
*   **Plot A (Time-Domain Waveforms)**: Displays raw vs enhanced waveforms with rolling amplitude envelopes.
*   **Plot B (Frequency-Domain Spectrograms)**: Heatmaps plotting time vs frequency in a styled dark-mode `magma` colormap, visually proving the erasure of background hum and white noise.
*   **Plot C (Average PSD Profile)**: A line chart plotting frequency (Hz) vs power (dB) showing the noise floor drop while speech harmonic peaks are preserved.

### 4. `app.py` (Module 4)
The front-end user interface built using Streamlit.
*   Interactive parameters (sliders for over-subtraction multiplier and spectral floor $\beta$).
*   **Synthetic Demo Generator**: Generates 5 seconds of harmonic speech (vocal fundamental + resonances) layered with a simulated 50Hz hum and broadband white noise.
*   **Interactive Trainer**: Allows training the model live in the browser using the backpropagation loop!

---

## 🧠 How to Train the AI Model

The application implements a cutting-edge **hybrid Deep Learning + differentiable DSP architecture**. The GRU neural network does not just run static predictions; it is trained end-to-end to minimize the loss in the enhanced audio magnitude spectrum!

### 1. Differentiable DSP Backpropagation (Self-Supervised / Supervised)
Because the power spectral subtraction equations, maximum bounds, and square root operations are written using fully differentiable PyTorch tensor math, PyTorch can track gradients through the physical equations.

During training, we:
1. Compute the noisy magnitude spectrum $|Y_t(f)|$ and clean magnitude spectrum $|X_t(f)|$.
2. Feed the 24-D Bark scale energy features of the noisy signal $B$ into the GRU model.
3. The model outputs a sequence of over-subtraction scalars $\alpha_t \in [0.0, 2.5]$.
4. Perform spectral subtraction on PyTorch tensors in the linear magnitude domain with recurrent temporal IIR smoothing to yield enhanced magnitude spectrum $|S_t(f)|$:
   $$|S_t(f)| = \text{IIR\_Smooth}\left( \max\left(|Y_t(f)| - \alpha_t \cdot (W_{\text{linear}} \cdot |D(f)|), \beta \cdot |Y_t(f)|\right) \right)$$
5. Compute the **Mean Squared Error (MSE)** loss:
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{T \cdot F} \sum_{t=1}^T \sum_{f=1}^F \left( |S_t(f)| - |X_t(f)| \right)^2$$
6. Run `loss.backward()`. The optimizer adjusts the GRU's recurrent and linear weights so that it dynamically outputs larger $\alpha_t$ values during noise-only frames and smaller $\alpha_t$ values (near 0) during speech frames to preserve vocal details.

### 2. Interactive Training in the Web App
1. Launch the app and click **"Generate Synthetic Demo"**.
2. Because the demo is synthetic, the backend knows the exact clean signal.
3. Select the desired number of **Training Epochs** (default: 25).
4. Click **"Train Differentiable GRU Model"**.
5. Watch the MSE loss decrease epoch-by-epoch on the rendered loss plot! Once complete, the enhanced audio is automatically regenerated using your newly optimized neural controller!

### 3. Scaling up to custom dataset files
To train this model on custom real-world speech files:
1. Compile a dataset of clean speech WAV files and a dataset of corresponding noise files (e.g. car noise, office chatter, street rumble).
2. Synthetically mix them to create matching noisy-clean training pairs: `noisy_audio = clean_audio + noise`.
3. In a custom script, load the pairs and run them through `dsp_engine` to get:
   *   Noisy Bark features `noisy_bark` (input)
   *   Noisy linear magnitudes `Y_mags` (input)
   *   Clean linear magnitudes `X_mags` (target)
4. Train the model using the `train_model()` function inside `neural_controller.py` in batches:
   ```python
   import torch
   from neural_controller import LightweightNeuralController, train_model
   from dsp_engine import DSPEngine
   
   # Initialize
   model = LightweightNeuralController()
   dsp = DSPEngine()
   
   # Load your custom WAVs
   noisy_audio = dsp.load_audio("custom_noisy_speech.wav")
   clean_audio = dsp.load_audio("custom_clean_speech.wav")
   
   # Extract features
   frames_noisy = dsp.time_domain_framing(noisy_audio)
   frames_clean = dsp.time_domain_framing(clean_audio)
   
   Y_mags, _ = dsp.compute_stft(frames_noisy)
   X_mags, _ = dsp.compute_stft(frames_clean)
   
   noisy_bark = dsp.project_linear_to_bark(Y_mags)
   clean_bark = dsp.project_linear_to_bark(X_mags)
   
   # Extract baseline noise profile (e.g. first 5 frames of silence)
   D_power = np.mean(np.square(Y_mags[:5]), axis=0)
   
   # Run the optimization loop!
   losses = train_model(
       model=model,
       noisy_audio_features=noisy_bark,
       clean_audio_features=clean_bark,
       noisy_linear_mags=Y_mags,
       clean_linear_mags=X_mags,
       D_mag=D_mag,
       W_linear=dsp.W_linear,
       beta=0.01,
       epochs=30,
       lr=0.01
   )
   
   # Save your trained PyTorch weights!
   torch.save(model.state_dict(), "trained_gru_controller.pth")
