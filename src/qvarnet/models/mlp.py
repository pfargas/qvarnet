import torch
from torch import nn


class MLP(nn.Module):
    """
    MLP class for creating a multi-layer perceptron.
    """

    def __init__(self, layer_dims = [1,60,60,1], activation: str = "tanh", pretrained: bool = False, pretrained_path: str = None):
        super().__init__()
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.hidden_dims = layer_dims[1:-1]
        self.activation = activation

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # Adding activation function for all layers except the last lasyer
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        
        
        # https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
        if pretrained and pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        """
        return self.model(x)
    
    def __repr__(self):
        """
        String representation of the MLP.
        """
        return f"MLP(input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_dims={self.hidden_dims}, activation={self.activation})"
        