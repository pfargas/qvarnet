import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Replicate your setup more closely
class MLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # No activation on last layer
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class HarmonicOscillator:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, x):
        # This is the CRITICAL part - let's debug this carefully
        psi = self.model(x)
        
        # First derivative
        dpsi_dx = torch.autograd.grad(
            outputs=psi,
            inputs=x,
            grad_outputs=torch.ones_like(psi),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative
        d2psi_dx2 = torch.autograd.grad(
            outputs=dpsi_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Local energy formula: E_L = (-1/2 * d2psi/dx2 + 1/2 * x^2 * psi) / psi
        kinetic = -0.5 * d2psi_dx2 / (psi + 1e-10)  # Add small epsilon for stability
        potential = 0.5 * x**2
        
        return kinetic + potential

# Simple Metropolis sampler for debugging
def simple_metropolis_sample(model, n_samples, step_size=0.5, burn_in=1000):
    """Simple metropolis sampler to test if the issue is in your sampler"""
    
    samples = []
    x = torch.tensor([0.0], device=device, requires_grad=True)
    
    accepted = 0
    
    for i in range(burn_in + n_samples):
        # Propose new position
        x_new = x + step_size * (torch.randn_like(x) - 0.5)
        
        # Compute acceptance probability
        with torch.no_grad():
            psi_old = model(x)
            psi_new = model(x_new)
            
            # Avoid division by zero
            if torch.abs(psi_old) < 1e-10:
                accept_prob = 1.0
            else:
                accept_prob = torch.min(torch.tensor(1.0), (psi_new / psi_old)**2)
        
        # Accept or reject
        if torch.rand(1).to(device) < torch.tensor(accept_prob).to(device):
            x = x_new.detach().clone()
            x.requires_grad = True
            accepted += 1
        
        # Store sample after burn-in
        if i >= burn_in:
            samples.append(x.detach().clone())
    
    acceptance_rate = accepted / (burn_in + n_samples)
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    return torch.stack(samples).requires_grad_(True)

# Test function to isolate the issue
def test_vmc_components():
    print("=== TESTING VMC COMPONENTS ===")
    
    # Create model with your architecture
    model = MLP([1, 2, 1]).to(device)
    
    # Initialize with small weights
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)
    
    hamiltonian = HarmonicOscillator(model)
    
    # Test 1: Check if model produces reasonable outputs
    x_test = torch.linspace(-3, 3, 100).view(-1, 1).to(device)
    x_test.requires_grad = True
    
    with torch.no_grad():
        psi_test = model(x_test)
        
    print(f"Model output range: [{psi_test.min().item():.6f}, {psi_test.max().item():.6f}]")
    print(f"Model output std: {psi_test.std().item():.6f}")
    
    # Test 2: Check Hamiltonian computation
    try:
        local_energy_test = hamiltonian(x_test)
        print(f"Local energy mean: {local_energy_test.mean().item():.6f}")
        print(f"Local energy std: {local_energy_test.std().item():.6f}")
        
        if torch.isnan(local_energy_test).any():
            print("ERROR: NaN in local energy!")
            return False
            
    except Exception as e:
        print(f"ERROR in Hamiltonian: {e}")
        return False
    
    # Test 3: Check gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for test_step in range(5):
        optimizer.zero_grad()
        
        # Simple samples for testing
        samples = torch.randn(1000, 1, device=device, requires_grad=True)
        
        local_energy = hamiltonian(samples)
        loss = local_energy.mean()
        
        loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
        
        print(f"Test step {test_step}: Loss = {loss.item():.6f}, Grad norm = {total_grad_norm:.6f}")
        
        if total_grad_norm < 1e-8:
            print("WARNING: Gradients are vanishing!")
            
        optimizer.step()
    
    return True

# Main VMC with debugging
def debug_vmc():
    print("\n=== MAIN VMC DEBUG ===")
    
    # Test components first
    if not test_vmc_components():
        print("Component test failed!")
        return
    
    # Parameters
    N_SAMPLES = 5000
    EPOCHS = 2000
    LEARNING_RATE = 1e-4
    
    # Model
    model = MLP([1, 2, 1]).to(device)
    
    # Initialize
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)
    
    hamiltonian = HarmonicOscillator(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    energy_history = []
    grad_norms = []
    psi_stds = []
    
    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        
        # CRITICAL: Use simple sampling first, then replace with your sampler
        # samples = simple_metropolis_sample(model, N_SAMPLES)
        
        # Or use your sampler (replace this):
        # x0 = torch.tensor([0.0], device=device)
        # samples = your_sampler(x0, method="parallel", n_walkers=N_SAMPLES)
        
        # For now, let's use random sampling to isolate the issue
        # samples = torch.randn(N_SAMPLES, 1, device=device, requires_grad=True)
        samples = simple_metropolis_sample(model, N_SAMPLES)
        
        # Check wavefunction
        psi_values = model(samples)
        psi_std = psi_values.std()
        psi_stds.append(psi_std.item())
        
        # Compute local energy
        local_energy = hamiltonian(samples)
        energy_mean = local_energy.mean()
        
        # Check for numerical issues
        if torch.isnan(energy_mean) or torch.isinf(energy_mean):
            print(f"Numerical issues at epoch {epoch}")
            break
        
        # VMC loss
        loss = energy_mean
        loss.backward()
        
        # Check gradients
        total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        grad_norms.append(total_grad_norm)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        energy_history.append(energy_mean.item())
        
        # Debugging output
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Energy = {energy_mean.item():.6f}, Psi_std = {psi_std.item():.6f}, Grad_norm = {total_grad_norm:.6f}")
    
    return model, energy_history, grad_norms, psi_stds

# Run debug
model, energy_history, grad_norms, psi_stds = debug_vmc()

# Plot comprehensive results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Energy history
axes[0, 0].plot(energy_history)
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='Analytical (0.5)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Energy')
axes[0, 0].set_title('Energy History')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Gradient norms
axes[0, 1].plot(grad_norms)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Gradient Norm')
axes[0, 1].set_title('Gradient Norms')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True)

# Psi standard deviation
axes[1, 0].plot(psi_stds)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Psi Standard Deviation')
axes[1, 0].set_title('Wavefunction Variability')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True)

# Final wavefunction
x_plot = torch.linspace(-3, 3, 1000).view(-1, 1).to(device)
with torch.no_grad():
    psi_plot = model(x_plot)
    psi_squared = psi_plot**2

axes[1, 1].plot(x_plot.cpu().numpy(), psi_squared.cpu().numpy(), 'b-', label='Learned |ψ|²')
analytical = np.exp(-0.5 * x_plot.cpu().numpy()**2) / (np.pi**0.25)
axes[1, 1].plot(x_plot.cpu().numpy(), analytical**2, 'r--', label='Analytical |ψ|²')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('|ψ(x)|²')
axes[1, 1].set_title('Final Wavefunction')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

print(f"\nFinal energy: {energy_history[-1]:.6f}")
print(f"Final psi_std: {psi_stds[-1]:.6f}")
print(f"Final grad_norm: {grad_norms[-1]:.6f}")

# CRITICAL QUESTIONS:
print("\n=== CRITICAL DEBUGGING QUESTIONS ===")
print("1. Does this version work with random sampling?")
print("2. If yes, replace random sampling with your MetropolisHastingsSampler")
print("3. If no, the issue is in the Hamiltonian or model setup")
print("4. Are you using any custom modifications to the loss or gradients?")
print("5. Are you doing anything with the samples before passing to Hamiltonian?")

# Next steps
print("\n=== NEXT STEPS ===")
print("1. Run this version - it should work with random sampling")
print("2. Replace the random sampling line with your actual sampler")
print("3. If it breaks, the issue is in your sampler")
print("4. If it still works, the issue is elsewhere in your code")