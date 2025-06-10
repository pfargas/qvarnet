import torch
from torch import nn
import time

class MetropolisHastingsSampler(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 n_samples: int = 1000, 
                 step_size: float = 0.1, 
                 burn_in: int = 100, 
                 is_wf: bool = True, 
                 batches: int = 1, 
                 is_batched: bool = False
                 ):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.step_size = step_size
        self.burn_in = burn_in
        self.is_wf = is_wf
        self.batches = batches
        self.values_per_batch = n_samples // batches
        self.is_batched = is_batched
        
        # Statistics tracking
        self.accepted_moves = 0
        self.total_moves = 0
        
        if n_samples % batches != 0:
            print("n_samples are not divisible by batches \n")
            print(f"Setting n_samples to {self.values_per_batch * batches} \n")
            self.n_samples = self.values_per_batch * batches

    def _evaluate_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized probability evaluation with batching"""
        if self.is_wf:
            # Assuming model returns [batch_size, ...], take appropriate slice
            model_output = self.model(x)
            if model_output.dim() > 1:
                return model_output.pow(2)[:, 0] if model_output.shape[1] > 1 else model_output.pow(2).squeeze()
            else:
                return model_output.pow(2)
        else:
            # Assume Boltzmann distribution
            return torch.exp(-self.model(x))

    def _mh_step_vectorized(self, x: torch.Tensor, n_walkers: int = 1) -> torch.Tensor:
        """Vectorized MH step for multiple walkers simultaneously"""
        
        # x has to have shape [n_walkers, dimensions]
        assert x.shape[0] == n_walkers, "x must have shape [n_walkers, dimensions]"
        
        # Generate proposals for all walkers at once
        x_new = x + torch.randn_like(x) * self.step_size
        
        # Evaluate probabilities in batch
        p_old = self._evaluate_probability(x)
        p_new = self._evaluate_probability(x_new)
        
        # Compute acceptance ratios
        acceptance_ratio = (p_new / (p_old + 1e-12)).clamp(max=1.0)
        
        # Accept/reject moves
        accept_mask = torch.rand_like(acceptance_ratio) < acceptance_ratio
        
        # Update positions
        if x.dim() == 1:
            accept_mask = accept_mask.unsqueeze(-1)
        elif x.dim() == 2 and accept_mask.dim() == 1:
            accept_mask = accept_mask.unsqueeze(-1)
            
        x_updated = torch.where(accept_mask, x_new, x)
        
        # Update statistics
        self.accepted_moves += accept_mask.sum().item()
        self.total_moves += accept_mask.numel()
        
        return x_updated

    def _mh_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single walker MH step (for compatibility)"""
        return self._mh_step_vectorized(x)

    def _mh_parallel_walkers(self, x0: torch.Tensor, n_walkers: int = 32) -> torch.Tensor:
        """Run multiple independent walkers in parallel"""
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        # Initialize multiple walkers
        x = x0.repeat(n_walkers, 1) + torch.randn(n_walkers, x0.shape[1], device=x0.device) * 0.1
        
        samples_per_walker = self.n_samples // n_walkers
        # if self.n_samples % n_walkers != 0:
        #     print("n_samples is not divisible by n_walkers, adjusting samples_per_walker.")
        #     samples_per_walker += 1
        samples = torch.zeros((n_walkers, samples_per_walker, x0.shape[1]), device=x0.device)
        
        with torch.no_grad():
            # Burn-in phase
            for _ in range(self.burn_in):
                x = self._mh_step_vectorized(x, n_walkers)
            
            # Sampling phase
            for i in range(samples_per_walker):
                x = self._mh_step_vectorized(x, n_walkers)
                samples[:, i] = x
        
        # Reshape to [total_samples, dimensions]
        return samples.view(-1, x0.shape[1])

    def _mh_no_batch_optimized(self, x0: torch.Tensor) -> torch.Tensor:
        """Optimized single-chain sampling"""
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        # Pre-allocate memory
        samples = torch.zeros((self.n_samples, x0.shape[1]), device=x0.device, dtype=x0.dtype)
        x = x0.clone()
        
        with torch.no_grad():
            # Burn-in phase
            for _ in range(self.burn_in):
                x = self._mh_step(x)
            
            # Sampling phase with reduced overhead
            for i in range(self.n_samples):
                x = self._mh_step(x)
                samples[i] = x.squeeze(0) if x.dim() > 1 else x
        
        return samples

    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate"""
        if self.total_moves == 0:
            return 0.0
        return self.accepted_moves / self.total_moves

    def reset_statistics(self):
        """Reset acceptance rate statistics"""
        self.accepted_moves = 0
        self.total_moves = 0

    def tune_step_size(self, x0: torch.Tensor, target_rate: float = 0.5, n_tune: int = 100):
        """Auto-tune step size to achieve target acceptance rate"""
        print(f"Tuning step size (target acceptance rate: {target_rate:.2f})")
        
        for iteration in range(10):  # Max tuning iterations
            self.reset_statistics()
            x = x0.clone()
            
            # Run short sampling to measure acceptance rate
            with torch.no_grad():
                for _ in range(n_tune):
                    x = self._mh_step(x)
            
            current_rate = self.get_acceptance_rate()
            print(f"Iteration {iteration + 1}: step_size={self.step_size:.4f}, acceptance_rate={current_rate:.3f}")
            
            if abs(current_rate - target_rate) < 0.05:  # Close enough
                break
            
            # Adjust step size
            if current_rate > target_rate:
                self.step_size *= 1.1  # Increase step size
            else:
                self.step_size *= 0.9  # Decrease step size
        
        print(f"Final step_size: {self.step_size:.4f}")
        self.reset_statistics()

    def forward(self, x0: torch.Tensor, method: str = 'parallel', n_walkers: int = 32) -> torch.Tensor:
        """
        Forward pass with different sampling methods
        
        Args:
            x0: Initial configuration
            method: 'single', 'parallel', 'block', or 'optimized'
            n_walkers: Number of parallel walkers (for parallel method)
        """
        if method == 'parallel' and n_walkers > 1:
            return self._mh_parallel_walkers(x0, n_walkers)
        elif method == 'no_batch_optimized':
            return self._mh_no_batch_optimized(x0)
        else:
            return self._mh_no_batch_optimized(x0)  # Default to optimized single chain


# Example usage and performance comparison
def benchmark_sampler(model, x0, n_samples=1000):
    """Benchmark different sampling methods"""
    
    sampler = MetropolisHastingsSampler(model, n_samples=n_samples, step_size=0.1)
    
    print("Benchmarking different methods:")
    print("-" * 50)
    
    methods = [
        ('optimized', {}),
        ('parallel', {'n_walkers': 16}),
        ('parallel', {'n_walkers': 32}),
        ('block', {})
    ]
    
    results = {}
    
    for method_name, kwargs in methods:
        start_time = time.time()
        samples = sampler.forward(x0, method=method_name, **kwargs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        acceptance_rate = sampler.get_acceptance_rate()
        
        print(f"{method_name:12s}: {elapsed:.3f}s, acceptance: {acceptance_rate:.3f}, shape: {samples.shape}")
        results[method_name] = {
            'time': elapsed,
            'acceptance_rate': acceptance_rate,
            'samples': samples
        }
        
        sampler.reset_statistics()
    
    return results