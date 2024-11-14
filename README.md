# fusion
Example code for diffusion models.


## Diffusion models

* **Naive diffusion** A simple implementation of the basic idea behind diffusion. Requires very little mathematics.
* **DDPM** The canonical diffusion model, based on carefully controlled Gaussian noise.
* **DDIM** An alternative sampler for DDPMs, based on a different probabilistic model for the data, which allows us to sample from a DDPM model in fewer steps, without adapting the training method.
