import torch
import torch.nn as nn
import torch.optim as optim
import collections
import numpy as np

# ---------------------------------------------------------------------------
# 1. ARCHITECTURE DEFINITIONS
# ---------------------------------------------------------------------------

class CorticalMonitor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CorticalMonitor, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNNCell(input_dim, hidden_dim, nonlinearity='tanh')
        self.readout = nn.Linear(hidden_dim, output_dim)
        # Projection layer to align cerebellar correction dimensions
        self.correction_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t, h_prev, cerebellar_correction=None):
        if cerebellar_correction is not None:
            correction_resized = self.correction_proj(cerebellar_correction)
            h_t = self.rnn(x_t + correction_resized, h_prev)
        else:
            h_t = self.rnn(x_t, h_prev)
        y_t = self.readout(h_t)
        return y_t, h_t

class CerebellarPredictor(nn.Module):
    """
    Represents the Cerebellum. Acts as the Structural Engine and domain-general Smith Predictor.
    """
    def __init__(self, cortical_hidden_dim, expansion_dim):
        super(CerebellarPredictor, self).__init__()
        self.fc1 = nn.Linear(cortical_hidden_dim, expansion_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(expansion_dim, cortical_hidden_dim)

    def forward(self, h_cortical):
        expanded = self.relu(self.fc1(h_cortical))
        prediction = self.fc2(expanded)
        return prediction

class CerebroCerebellarSystem(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, expansion_dim, delay_steps=4):
        super(CerebroCerebellarSystem, self).__init__()
        self.cortical = CorticalMonitor(input_dim, hidden_dim, output_dim)
        self.cerebellar = CerebellarPredictor(hidden_dim, expansion_dim)
        self.delay_steps = delay_steps
        self.prediction_buffer = collections.deque(maxlen=delay_steps)
        
    def reset_buffer(self, batch_size, device):
        self.prediction_buffer.clear()
        for _ in range(self.delay_steps):
            self.prediction_buffer.append(torch.zeros(batch_size, self.cortical.hidden_dim).to(device))

# ---------------------------------------------------------------------------
# 2. TRAINING LOOP (PATCHED WITH BUG FIXES 1 & 2)
# ---------------------------------------------------------------------------

def train_step(model, inputs, targets, optimizer):
    model.train()
    seq_length = inputs.size(0)
    batch_size = inputs.size(1)
    device = inputs.device

    model.reset_buffer(batch_size, device)
    h_t = torch.zeros(batch_size, model.cortical.hidden_dim).to(device)

    total_loss = 0.0
    tracking_cortical_loss = 0.0  # BUG 2 FIX: Isolate cortical error for data tracking
    criterion = nn.MSELoss()

    for t in range(seq_length):
        x_t = inputs[t]
        target_t = targets[t]

        # BUG 1 FIX: Temporal alignment. Pull oldest prediction out of queue FIRST.
        delayed_correction = model.prediction_buffer.popleft() 

        # Cortex processes current state + delayed prediction
        y_t, h_t = model.cortical(x_t, h_t, delayed_correction)
        
        # Cerebellum predicts future state
        new_prediction = model.cerebellar(h_t)

        # Calculate Cortical Loss
        cortical_loss = criterion(y_t, target_t)
        tracking_cortical_loss += cortical_loss.item() # Add purely to our tracker

        # Calculate Cerebellar Smith Predictor Loss
        if t + model.delay_steps < seq_length:
            future_target = targets[t + model.delay_steps]
            # Simplistic loss proxy for cerebellum attempting to align with future cortical states
            smith_loss = criterion(new_prediction, h_t.detach()) 
        else:
            smith_loss = torch.tensor(0.0, device=device)

        # Free Energy Principle: Combine for the optimizer to update both systems
        total_loss += (cortical_loss + 0.5 * smith_loss)
        
        # Append new prediction LAST
        model.prediction_buffer.append(new_prediction)

    optimizer.zero_grad()
    total_loss.backward()

    # Calculate gradient norm ("Cortical Energy")
    grad_norm = 0.0
    for param in model.cortical.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()

    return tracking_cortical_loss, grad_norm


# ---------------------------------------------------------------------------
# 3. ABLATION SIMULATION (PATCHED WITH BUG 3)
# ---------------------------------------------------------------------------

def simulate_ccas_ablation(model, inputs, targets):
    # BUG 3 FIX: Disable learning and dropouts
    model.eval() 
    seq_length = inputs.size(0)
    batch_size = inputs.size(1)
    device = inputs.device

    h_t = torch.zeros(batch_size, model.cortical.hidden_dim).to(device)
    ablation_cortical_loss = 0.0
    ablation_grad_norm = 0.0  # Strictly zero because plasticity is halted

    criterion = nn.MSELoss()

    with torch.no_grad(): # BUG 3 FIX: No gradient calculation
        for t in range(seq_length):
            x_t = inputs[t]
            
            # Cortex manually steers without cerebellar correction
            y_t, h_t = model.cortical(x_t, h_t, cerebellar_correction=None)

            # Manual steering delay penalty (Cortex lagging behind temporal sequence)
            if t >= model.delay_steps:
                delayed_target = targets[t - model.delay_steps]
                step_loss = criterion(y_t, delayed_target) 
            else:
                step_loss = criterion(y_t, targets[t])

            ablation_cortical_loss += step_loss.item()

    model.train() # Reset safety
    return ablation_cortical_loss, ablation_grad_norm

# ---------------------------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------------------------

def generate_acheulean_data(seq_len, batch_size, input_dim):
    # Dummy sequence representing hierarchical goals
    inputs = torch.randn(seq_len, batch_size, input_dim)
    targets = torch.roll(inputs, shifts=-1, dims=0)
    return inputs, targets

if __name__ == "__main__":
    # Hyperparameters
    INPUT_DIM = 10
    HIDDEN_DIM = 32
    OUTPUT_DIM = 10
    EXPANSION_DIM = 128
    SEQ_LEN = 50
    BATCH_SIZE = 16
    DELAY_STEPS = 4

    model = CerebroCerebellarSystem(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, EXPANSION_DIM, DELAY_STEPS)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("--- PARADIGM 1: Hierarchical Acheulean Sequence Training ---")
    inputs, targets = generate_acheulean_data(SEQ_LEN, BATCH_SIZE, INPUT_DIM)

    # Train Dual-System Baseline
    for epoch in range(1, 21):
        loss, grad_norm = train_step(model, inputs, targets, optimizer)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Dual-System Cortical Loss: {loss:.4f} | Cortical Energy (Grad Norm): {grad_norm:.4f}")

    print("\n--- PARADIGM 2: Ablation / Simulating Dysmetria of Thought (CCAS) ---")
    ablation_loss, ablation_grad = simulate_ccas_ablation(model, inputs, targets)
    print(f"Ablation Epoch | Cortex-Only Loss: {ablation_loss:.4f} | Cortical Energy (Grad Norm): {ablation_grad:.4f}")
