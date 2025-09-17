# JAX notes on the side

## Gradient function

Imagine you have some function $f(\theta, x)$ and you want to compute $\nabla_\theta f(\theta, x)$. In JAX:

```python
import jax

params = [...]

grad_fn = jax.grad(fun(params, x), argnums=0)
grad_fn(params, x)
```
- `grad_fn` is a `function` with _the same number of arguments_ as the function passed inside (in this case `fun`)
  - The first argument is the function to take the derivative of
  - `argnums` accepts an iterable. Index of the argument to differentiate wrt. The argument **must** be an inexact type (float/complex)
- `jax.grad()` uses AD to return the function "derivative of function wrt the arguments"

After this, you can call the function as a normal function. This function can be vectorized with vmap as any other function. 

## Flax, NN and training pipeline

```python
import jax
import optax
from flax.training import train_state
from flax import linen as nn

def loss(input, params, model):
    loss_val = ...
    return loss_val

def loss_and_grads(params, batch, model):
    loss = jax.vmap(lambda x: loss(params, x, model))(batch)
    grads = ... # compute gradients with AD
    return loss, grads

@jax.jit
def training_step(state, batch):
    loss, grads = loss_and_grads(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train(n_steps, init_params, model, training_set, optimizer):
    state = train_state.TrainState.create(
        apply_fn=model_apply,
        params=init_params,
        tx=optimizer
    )

    for step in range(n_steps):
        batch = trainig_set[step]
        state, loss = training_step(state, batch)
    
    return state.params # (or state directly...)
```
- `loss`: the loss function of the problem. In our case, this would be the energy (mean of local energy)
- `loss_and_grads`: vectorized computation of the loss function and the gradients of it. In our case this would compute for all samples the energy and the **gradient of $\ln\psi(\equiv\nabla_\theta\psi)$. It is important to note that in our case, the jax.grad will be used to compute this.** _Of course, as in the same function we have $E_L(x)$ and $\nabla_\theta\ln\psi$, the output of the WHOLE function will be $E_L$ and the correct $\nabla_\theta E = 2\mathbb{E}_{x_i\sim \psi^2}\bigg[(E_L(x)-E)\nabla_\theta\ln\psi\bigg]$._
  - TL;DR: in `loss_and_grads` we compute in this order:
    1. Energy as the mean of the local energy
    2. gradient of the model ($\ln\psi$) with AD
    3. Apply the gradient to all batch (vectorized)
    4. Compute the correct gradient
    5. Return $E$ and $\nabla_\theta E$
- `training_step`: jitted function