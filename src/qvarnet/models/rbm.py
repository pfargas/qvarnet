import torch
from torch import nn

class RBM(nn.Module):
    """
    Restricted Boltzmann Machine (RBM) class for creating a RBM model.
    
    This class implements a Restricted Boltzmann Machine (RBM) with a specified number of visible and hidden units.
    
    Args:
        n_visible (int): Number of visible units.
        n_hidden (int): Number of hidden units.
        pretrained (bool): If True, load pretrained weights. Default is False.
        pretrained_path (str): Path to the pretrained weights. Default is None.
    """
    
    def __init__(self, n_visible: int, n_hidden: int, pretrained: bool = False, pretrained_path: str = None):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.b_v = nn.Parameter(torch.zeros(n_visible))
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path))
        
    def sample_h(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sample hidden units given visible units.
        
        Args:
            v (torch.Tensor): Visible units.
        
        Returns:
            torch.Tensor: Sampled hidden units.
        """
        h = torch.sigmoid(torch.matmul(v, self.W) + self.b_h)
        return h
    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given hidden units.
        
        Args:
            h (torch.Tensor): Hidden units.
            
        Returns:

            torch.Tensor: Sampled visible units.
        """
        v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b_v)
        return v
    
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RBM.
        
        Args:
            v (torch.Tensor): Visible units.
            
        Returns:
            torch.Tensor: Output of the RBM.
        """
        h = self.sample_h(v)
        v = self.sample_v(h)
        return v