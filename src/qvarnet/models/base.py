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

    @classmethod
    def from_config(cls, model_args: dict):
        """Instantiate this model from a model_args config dict.

        Every subclass must override this so that main.py can construct
        any model without knowing its specific constructor signature.

        Args:
            model_args: The 'model' section of the experiment config.

        Returns:
            An instance of the model.
        """
        raise NotImplementedError(f"{cls.__name__} must implement from_config().")

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        """Return the (batch_size, input_dim) shape used to initialise the sampler.

        Every subclass must override this so that main.py can determine
        the correct input shape without model-specific branching.

        Args:
            model_args: The 'model' section of the experiment config.
            batch_size: Training batch size.

        Returns:
            A (batch_size, input_dim) tuple.
        """
        raise NotImplementedError(f"{cls.__name__} must implement get_input_shape().")
