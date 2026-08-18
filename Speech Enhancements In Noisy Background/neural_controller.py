"""
MODULE 3: LIGHTWEIGHT NEURAL CONTROLLER (Member 3 Domain)
Defines the PyTorch lightweight GRU model and the differentiable DSP end-to-end training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LightweightNeuralController(nn.Module):
    def __init__(self, input_size=24, hidden_size=32, num_layers=1):
        super(LightweightNeuralController, self).__init__()
        # GRU tracks the slow-moving temporal noise floor across frames
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # Fully connected layer maps hidden state to a single continuous scalar output
        self.fc = nn.Linear(hidden_size, 1)
        # Sigmoid activation to bound the output, scaled by 1.5 to keep alpha in [0.0, 1.5]
        self.sigmoid = nn.Sigmoid()
        self.scale_factor = 2.5
        
        # Initialize weights such that it outputs sensible defaults (around 0.5 - 1.0)
        # and has a strong negative correlation with the overall energy (energy-dependent tracking)
        self._init_sensible_weights()

    def _init_sensible_weights(self):
        # We want the network to output lower values for high energy inputs (speech)
        # and higher values for lower energy inputs (noise/silence).
        with torch.no_grad():
            # Set input-to-hidden weights to have negative bias towards energy accumulation
            nn.init.xavier_uniform_(self.gru.weight_ih_l0)
            nn.init.orthogonal_(self.gru.weight_hh_l0)
            self.gru.bias_ih_l0.fill_(0.0)
            self.gru.bias_hh_l0.fill_(0.0)
            
            # FC layer weight: we initialize with negative values so high GRU activations
            # (which correspond to high energy) push the output lower.
            self.fc.weight.fill_(-0.1)
            self.fc.bias.fill_(0.2) # yields around alpha = 0.5 to 0.8 at startup

    def forward(self, x, h0=None):
        """
        Ingests the Bark energy vector frame-by-frame.
        
        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_len, 24)
            h0 (Tensor, optional): Initial hidden state of shape (num_layers, batch_size, hidden_size)
            
        Returns:
            alpha (Tensor): Bounded over-subtraction factor alpha_t of shape (batch_size, seq_len, 1)
            h_n (Tensor): Final hidden state of shape (num_layers, batch_size, hidden_size)
        """
        # x shape: (batch_size, seq_len, 24)
        out, h_n = self.gru(x, h0)
        # out shape: (batch_size, seq_len, hidden_size)
        
        # Map hidden state of each frame to scalar
        fc_out = self.fc(out)
        # fc_out shape: (batch_size, seq_len, 1)
        
        # Bounded between [0.0, 1.5]
        alpha = self.sigmoid(fc_out) * self.scale_factor
        return alpha, h_n

def differentiable_spectral_subtraction(Y_mag, alpha, D_mag, W_linear, beta=0.01):
    """
    Implements a fully differentiable, PyTorch-compatible linear magnitude spectral subtraction core
    with Wiener-style soft attenuation, 3-point frequency smoothing, and first-order IIR temporal smoothing.
    """
    batch_size, seq_len, num_bins = Y_mag.shape
    
    # Calculate weighted linear noise profile: shape (241,)
    weighted_noise = D_mag * W_linear
    # Add batch and sequence dimensions for broadcasting: shape (1, 1, 241)
    weighted_noise = weighted_noise.view(1, 1, -1)
    
    # Calculate Wiener-style soft attenuation factor H, bounded to floor of 0.15
    H = torch.maximum(1.0 - alpha * (weighted_noise / (Y_mag + 1e-6)), torch.tensor(beta, device=Y_mag.device))
    raw_clean = H * Y_mag
    
    # Apply active 3-point frequency-axis smoothing BEFORE temporal IIR smoothing
    # using 1D convolution with reflection padding to prevent phase tearing echo.
    x_reshaped = raw_clean.view(batch_size * seq_len, 1, num_bins)
    x_padded = torch.nn.functional.pad(x_reshaped, (1, 1), mode='reflect')
    kernel = torch.tensor([0.15, 0.7, 0.15], dtype=torch.float32, device=Y_mag.device).view(1, 1, 3)
    smoothed = torch.nn.functional.conv1d(x_padded, kernel)
    raw_clean_smooth = smoothed.view(batch_size, seq_len, num_bins)
    
    # Differentiable first-order IIR temporal smoothing loop across frames
    S_enhanced_list = []
    prev_clean = raw_clean_smooth[:, 0, :] # Initialize with first frame, shape (batch_size, 241)
    S_enhanced_list.append(prev_clean.unsqueeze(1))
    
    for t in range(1, seq_len):
        current_clean = raw_clean_smooth[:, t, :]
        smoothed_clean = 0.4 * prev_clean + 0.6 * current_clean
        S_enhanced_list.append(smoothed_clean.unsqueeze(1))
        prev_clean = smoothed_clean
        
    S_enhanced = torch.cat(S_enhanced_list, dim=1)
    return S_enhanced

def train_model(model, noisy_audio_features, clean_audio_features, 
                noisy_linear_mags, clean_linear_mags, 
                D_mag, W_linear, beta=0.01, epochs=30, lr=0.005):
    """
    Differentiable training loop. Minimizes the Mean Squared Error (MSE) 
    between the estimated clean magnitude and the true target clean magnitude.
    """
    # Convert numpy arrays to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Add batch dimension: (1, seq_len, feature_dim)
    x_bark = torch.tensor(noisy_audio_features, dtype=torch.float32).unsqueeze(0).to(device)
    Y_mag = torch.tensor(noisy_linear_mags, dtype=torch.float32).unsqueeze(0).to(device)
    X_clean = torch.tensor(clean_linear_mags, dtype=torch.float32).unsqueeze(0).to(device)
    D_m = torch.tensor(D_mag, dtype=torch.float32).to(device)
    W_lin = torch.tensor(W_linear, dtype=torch.float32).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Predict dynamic over-subtraction factor alpha (1, seq_len, 1)
        alpha, _ = model(x_bark)
        
        # Compute differentiable spectral subtraction
        S_enhanced = differentiable_spectral_subtraction(
            Y_mag=Y_mag,
            alpha=alpha,
            D_mag=D_m,
            W_linear=W_lin,
            beta=beta
        )
        
        # Loss: MSE between enhanced magnitude and true clean magnitude
        loss = criterion(S_enhanced, X_clean)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    return losses
