def init_weights(m):
    if isinstance(m, nn.Linear):
        # Small random weights - very important for VMC!
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def sampling_step(sampler, model, hamiltonian, optimizer, device):
    # Run sampler
    sampler.model = model
    x0 = torch.tensor([0.0], device=device)  # Initial point for the sampler
    samples = sampler(x0, method="parallel", n_walkers=(N_SAMPLES))
    samples.requires_grad = True
    
    # Compute the mean and std of the local energy
    hamiltonian.model = model
    local_energy = hamiltonian(samples)
    
    loss = local_energy.mean()
    
    # Compute gradients
    loss.squeeze().backward()

    optimizer.step()

    
    sampler.reset_statistics()
