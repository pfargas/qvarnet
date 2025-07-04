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
        if n_samples % batches != 0:
            print("n_samples are not divisible by batches \n")
            print(f"Setting n_samples to {self.values_per_batch * batches} \n")
            self.n_samples = self.values_per_batch * batches
            
        
    def _mh_step(self, x: torch.Tensor) -> torch.Tensor:
        x_new = x + torch.randn_like(x) * self.step_size

        if self.is_wf:
            start_time = time.time()
            p_new = self.model(x_new).pow(2)[0] # LOOK INTO THIS
            end_time = time.time()
            print(f"Model evaluation time: {end_time - start_time:.6f} seconds")
            # model returns [D, out]? if so, look into how to define this
            p_old = self.model(x).pow(2)[0]
        else:
            start_time = time.time()
            p_new = torch.exp(-self.model(x_new))
            end_time = time.time()
            print(f"Model evaluation time: {end_time - start_time:.6f} seconds")
            p_old = torch.exp(-self.model(x))

        start_time = time.time()
        acceptance_ratio = (p_new / (p_old + 1e-12)).clamp(max=1.0)
        accept = (torch.rand_like(acceptance_ratio) < acceptance_ratio).float().unsqueeze(-1)
        x = accept * x_new + (1 - accept) * x
        end_time = time.time()
        print(f"Acceptance ratio computation time: {end_time - start_time:.6f} seconds")
        return x
    
    def _mh_single_batch(self, x: torch.Tensor) -> torch.Tensor:
        samples_batch = torch.zeros((self.values_per_batch, x.shape[1]), device=x.device)
        for step in range(self.values_per_batch):
            x = self._mh_step(x)
            samples_batch[step] = x.view(1, -1).clone()
        samples_batch.requires_grad_(True)
        return samples_batch # shape [values_per_batch, D]
    
    def _mh_batched(self, x: torch.Tensor) -> torch.Tensor:
        samples = torch.zeros((self.batches, self.values_per_batch, x.shape[1]), device=x.device)
        x = x.clone().detach()
        with torch.no_grad():
            for i in range(self.batches): #parallelize this
                batch_samples = self._mh_single_batch(x)
                samples[i] = batch_samples # shape [values_per_batch, D]
        return None
    
    def _mh_no_batch(self, x0: torch.Tensor) -> torch.Tensor:
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)  # Make it shape [1, D]
        samples = torch.zeros((self.n_samples, x0.shape[1]), device=x0.device)
        x = x0.clone().detach()
        with torch.no_grad():
            for i in range(self.n_samples + self.burn_in):
                x = self._mh_step(x)
                if i >= self.burn_in:
                    samples[i - self.burn_in] = x.clone().view(1,-1) # force shape [1, D]

        samples.requires_grad_(True)
        return samples
    
    # def importance_sampling(self) -> torch.Tensor:
    #     """Importance sampling with wavefunction"""
        
    #     x0 = torch.rand((self.n_samples), device=self.model.device)  # Initialize with random samples
    #     x0 = x0.unsqueeze(-1)  # Ensure shape is [n_samples, 1]
    #     mask = 
        
        
        
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if self.is_batched:
            print("Not implemented yet\n")
            print("Using no batching instead\n")
            return self._mh_no_batch(x0)
        else:
            return self._mh_no_batch(x0)