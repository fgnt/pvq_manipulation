
"""
Implementation of probability paths for flow matching, taken from the paper "FLOW MATCHING FOR GENERATIVE MODELING"
https://arxiv.org/pdf/2210.02747
"""

import torch


class Optimal_Transport(torch.nn.Module):
    def __init__(self, sigma_min=1.e-17):
        super().__init__()
        self.sigma_min = sigma_min

    def mean(self, x, t):
        return t * x
    
    def sigma(self, x, t):
        return 1 - (1-self.sigma_min) * t
    
    def mean_derivative(self, x, t):
        return x
    
    def sigma_derivative(self, x, t):
        return -(1-self.sigma_min)
    
class DiffusionVarianceExploding(torch.nn.Module):
    def __init__(self, sigma_min=0.01, sigma_max=10):
        super().__init__()
        self.sigma_min = torch.tensor(sigma_min)
        self.sigma_max = torch.tensor(sigma_max)

    def mean(self, x, t):
        return x

    def sigma(self, x, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (1 - t)

    def mean_derivative(self, x, t):
        return torch.zeros_like(x)

    def sigma_derivative(self, x, t):
        sigma = self.sigma(x, t)
        return -sigma * torch.log(self.sigma_max / self.sigma_min)
    

class DiffusionVariancePreserving(torch.nn.Module):

    def __init__(self, beta_min=0.1, beta_max=50.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def mean(self, x_1, t):
        """ μ_t(x_1) = α(1-t) * x_1 """
        return self.alpha(1 - t) * x_1

    def sigma(self, x_1, t):
        """ σ_t(x_1) = sqrt(1 - α(1-t)²) """
        return torch.sqrt(torch.clamp(1 - self.alpha(1 - t)**2, min=1e-6))
    
    def mean_derivative(self, x_1, t):
        return - self.alpha_derivative(1 - t) * x_1

    def sigma_derivative(self, x_1, t):
        a = self.alpha(1 - t)
        a_dot = - self.alpha_derivative(1 - t)
        denom = torch.sqrt(torch.clamp(1 - a**2, min=1e-6))
        return -(a * a_dot) / denom

    def beta(self, t):
        """
        noise schedule β(t)
        """                                                    
        return self.beta_min + t * (self.beta_max - self.beta_min)


    def T(self, t):
        """
        T(t) = ∫₀ᵗ β(s) ds        
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
    
    def T_derivative(self, t):
        return self.beta(t)

    def alpha(self, t):
        return torch.exp(-0.5 * self.T(t))

    def alpha_derivative(self, t):
        return -0.5 * self.T_derivative(t) * self.alpha(t)