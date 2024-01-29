"""
This file contains additional distributions.
"""


import torch
import torch.distributions as dist

class ClippedGaussian1D(dist.Distribution):
    def __init__(self, mus: torch.Tensor, sigmas: torch.Tensor, 
                 clip_min: torch.Tensor = torch.tensor([0]), 
                 clip_max: torch.Tensor = torch.tensor([1])):
        self.mus = mus
        self.sigmas = sigmas
        self.clip_min = clip_min
        self.clip_max = clip_max

    def sample(self, sample_shape=torch.Size()):
        samples = torch.stack([dist.Normal(mu, sigma).sample(sample_shape) for mu, sigma in zip(self.mus, self.sigmas)], dim=-1)
        return torch.clamp(samples, self.clip_min, self.clip_max)
    
    def log_prob(self, value):
        raise NotImplementedError("ClippedGaussian log_prob not implemented")

class ClippedGaussian(dist.Distribution):
    def __init__(self, mus: torch.Tensor, sigmas: torch.Tensor, 
                 clip_min: torch.Tensor = torch.tensor([0]), 
                 clip_max: torch.Tensor = torch.tensor([1])):
        self.mus = mus
        self.sigmas = sigmas
        self.clip_min = clip_min
        self.clip_max = clip_max

    def sample(self, sample_shape=torch.Size()):
        samples = torch.stack([dist.MultivariateNormal(mu, sigma).sample(sample_shape) for mu, sigma in zip(self.mus, self.sigmas)], dim=0)
        return torch.clamp(samples, self.clip_min, self.clip_max)
    
    def log_prob(self, value):
        raise NotImplementedError("ClippedGaussian log_prob not implemented")
    
if __name__ == "__main__":
    mus = torch.arange(10).view(5, 2).float() * 0.1
    sigmas = torch.eye(mus.shape[-1]).view(1, mus.shape[-1], mus.shape[-1]).repeat(mus.shape[0], 1, 1) * 0.0001
    print(mus)
    print(sigmas)
    cg = ClippedGaussian(mus, sigmas)
    print(cg.sample())
    # print(cg.sample())

