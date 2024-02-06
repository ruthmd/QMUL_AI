#### This is a simple implementation to generate adversarial example of MNIST dataset and demonstrates the FSGM method to do the same.

$$
adv\_x = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$


- `adv_x`: Adversarial image.
- `x`: Original input image.
- `y`: Original input label.
- `ε`: Multiplier to ensure the perturbations are small.
- `θ`: Model parameters.
- `J`: Loss.

