from flax import linen as nn


class BaseModel(nn.Module):
    """Base class for all models in QVarNet."""

    def __call__(self, x):
        """Forward pass of the model.

        Args:
            x: Input data.

        Returns:
            The output of the model.
        """
        raise NotImplementedError("Subclasses must implement __call__ method.")

    def build_from_params(self, params):
        """Build the model from a set of parameters.

        Args:
            params: A dictionary of parameters to build the model.
        """
        raise NotImplementedError("Subclasses must implement build_from_params method.")
