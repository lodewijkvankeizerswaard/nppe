"""
This file contains the code to generate a Lotka-Volterra dataset. 
"""

import torch
import torch.distributions as dist
import numpy as np
import os

from utils import Logger
from lotka_volterra_utils import load_debug_dataset, solve_simulation_batch
from settings import ExperimentSettings
from distributions import ClippedGaussian1D

def get_data(settings: ExperimentSettings,
             logger: Logger | None = None
             ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    # Load debug data if specified
    if  settings.bn['load_debug_data']:
        raw_train = load_debug_dataset('train')
        raw_val = load_debug_dataset('val')
        raw_test = load_debug_dataset('test')
    generate_data = True

    # Check if the data is already generated
    data_dir = os.path.join("data", 
                            settings.bn['p_alpha'], 
                            settings.bn['p_theta_alpha'], 
                            f"{settings.bn['train_I']}-\
                              {settings.bn['train_J']}-\
                              {settings.bn['val_I']}-\
                              {settings.bn['val_J']}-\
                              {settings.bn['test_I']}-\
                              {settings.bn['test_J']}".replace(' ', ''), 
                            str(settings.training["seed"]))
    if os.path.exists(data_dir) and not settings.bn['load_debug_data']:
        if logger:
            logger.log("Found generated data")
        raw_train = torch.load(os.path.join(data_dir, "train.pt"))
        raw_val = torch.load(os.path.join(data_dir, "val.pt"))
        raw_test = torch.load(os.path.join(data_dir, "test.pt"))

        generate_data = False
        if raw_train[2].shape[0] < settings.bn["train_N"] or \
           raw_val[2].shape[0] < settings.bn["val_N"] or \
           raw_test[2].shape[0] < settings.bn["test_N"]:
            # Not enough data: regenerating
            if logger:
                logger.log("Not enough data, regenerating")
            generate_data = True

        elif raw_train[2].shape[0] == settings.bn["train_N"] and \
             raw_val[2].shape[0] == settings.bn["val_N"] and \
             raw_test[2].shape[0] == settings.bn["test_N"]:
            if logger:
                logger.log("Loading generated data")
        
        else:
            # Even though "overgeneration" in a previous run will result in different data
            # We will not regenerate the data if more is available then needed
            raw_train[0] = raw_train[0][:settings.bn['train_N']]
            raw_train[1] = raw_train[1][:settings.bn['train_N']]
            raw_train[2] = raw_train[2][:settings.bn['train_N']]
            raw_val[0] = raw_val[0][:settings.bn['train_N']]
            raw_val[1] = raw_val[1][:settings.bn['train_N']]
            raw_val[2] = raw_val[2][:settings.bn['train_N']]
            raw_test[0] = raw_test[0][:settings.bn['train_N']]
            raw_test[1] = raw_test[1][:settings.bn['train_N']]
            raw_test[2] = raw_test[2][:settings.bn['train_N']]

        
    # Generate datasets if needed
    # if not settings.bn['load_debug_data'] and not data_loaded:
    if generate_data and not settings.bn['load_debug_data']:
        if logger:
            logger.log("Generating data")
        lv = BNLotkaVolterra(settings)
        raw_train = lv.generate_dataset(N=settings.bn['train_N'], 
                                          I=settings.bn['train_I'], 
                                          J=settings.bn['train_J'], 
                                          suffix='train',
                                          logger=logger)
        raw_val = lv.generate_dataset(N=settings.bn['val_N'],
                                        I=settings.bn['val_I'],
                                        J=settings.bn['val_J'],
                                        suffix='val',
                                        logger=logger)
        raw_test = lv.generate_dataset(N=settings.bn['test_N'],
                                         I=settings.bn['test_I'],
                                         J=settings.bn['test_J'],
                                         suffix='test',
                                         logger=logger)
    # Save the data if we (re)generated
    if generate_data and settings.bn['save_data'] and logger != None and not settings.bn['load_debug_data']:
        logger.log("Saving generated data")
        logger.save_data('train', raw_train)
        logger.save_data('val', raw_val)
        logger.save_data('test', raw_test)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        torch.save(raw_train, os.path.join(data_dir, "train.pt"))
        torch.save(raw_val, os.path.join(data_dir, "val.pt"))
        torch.save(raw_test, os.path.join(data_dir, "test.pt"))

    # Group up the data
    train = BNLotkaVolterra.group_up_data(*raw_train, flatten=True)
    val = BNLotkaVolterra.group_up_data(*raw_val, flatten=True)
    test = BNLotkaVolterra.group_up_data(*raw_test, flatten=True)

    return train, val, test

class BNLotkaVolterra:
    def __init__(self, settings: ExperimentSettings):
        self.bn = settings.bn
        self.nr_threads = settings.training['nr_threads']
        self.settings = settings
        
        # First we infer the shapes of alpha, and x (theta is always (6,))
        self.alpha_shape = (12,) if settings.bn['p_theta_alpha'] == 'gaussian' else (6,)
        self.x_shape = (2, settings.bn['M'])
        self.alpha_parameter_count = np.prod(self.alpha_shape)

    def _p_alpha(self) -> dist.Distribution:
        """
        Returns a distribution over the parameters of $p(\theta|\alpha)$. This currently does not
        factor in the support over the parameters of $p(\theta|\alpha)$; i.e. it assumes all parameters
        are in [0,1].
        
        Returns:
        - torch.tensor: A distribution over the LV parameters.
        """

        if self.bn['p_alpha'] == 'uniform':
            lower_bound = torch.zeros(self.alpha_parameter_count)
            upper_bound = torch.ones(self.alpha_parameter_count)
            return dist.Uniform(lower_bound, upper_bound)

        elif self.bn['p_alpha'] == 'continuousbern':
            s = self.bn['p_alpha_args']
            if 'lambda_alpha' in s:
                probs = torch.tensor([s['lambda_alpha']]).repeat(self.alpha_parameter_count)
            else:
                parameter_names = [f"lambda_alpha_{c}" for c in range(self.alpha_parameter_count)]
                probs = torch.tensor([s[p] for p in parameter_names])
            return dist.continuous_bernoulli.ContinuousBernoulli(probs=probs)

        elif self.bn['p_alpha'] == 'gaussian':
            s = self.bn['p_alpha_args']
            if 'mu_alpha' in s:
                mu = torch.tensor([s['mu_alpha']]).repeat(self.alpha_parameter_count)
                sigma = torch.tensor([s['sigma_alpha']]).repeat(self.alpha_parameter_count)
                return ClippedGaussian1D(mus=mu, sigmas=sigma)

            parameter_names_means = [f"mu_alpha_{c}" for c in range(self.alpha_parameter_count)]
            parameter_names_sigmas = [f"sigma_alpha_{c}" for c in range(self.alpha_parameter_count)]
            loc = torch.tensor([s[p] for p in parameter_names_means])
            scale = torch.tensor([s[p] for p in parameter_names_sigmas])
            return ClippedGaussian1D(mus=loc, sigmas=scale)
        else:
            raise ValueError(f'Unknown p_alpha: {self.bn["p_alpha"]}')
    
    def _p_theta_alpha(self, alpha: torch.Tensor) -> dist.Distribution:
        """
        Computes the probability distribution over the model parameters theta given the hyperparameters alpha.

        Args:
            alpha (torch.Tensor): The hyperparameters alpha.

        Returns:
            dist.Distribution: The probability distribution over the model parameters theta.
        """
        if self.bn['p_theta_alpha_args'] != {}:
            print("Warning: p_theta_alpha_args is not empty, but it is not used for the LV model.")

        if self.bn['p_theta_alpha'] == 'continuousbern':
            return dist.continuous_bernoulli.ContinuousBernoulli(probs=alpha)
        elif self.bn['p_theta_alpha'] == 'gaussian':
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(0)
            mus = alpha[:, :6]
            sigmas = alpha[:, 6:] + 0.0001  # Add a small value to prevent sigmas from being 0
            return ClippedGaussian1D(mus=mus, sigmas=sigmas)
        elif self.bn['p_theta_alpha'] == 'gaussian-means':
            return ClippedGaussian1D(mus=alpha, sigmas=torch.ones(6)*0.001)
        else:
            raise ValueError(f'Unknown p_theta_alpha: {self.bn["p_theta_alpha"]}')


    def _p_x_theta(self, theta):
        """ This is the LV simulation. """
        return solve_simulation_batch(theta, self.bn['M'], self.bn['dt'], processes=self.nr_threads)

    def generate_dataset(self, N: int, I: int, J: int, suffix: str = None, logger: Logger | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a dataset for the Lotka-Volterra model with the given parameters.

        Args:
            N (int): Number of alpha samples to generate.
            I (int): Number of initial conditions to sample for each alpha.
            J (int): Number of noise samples to generate for each initial condition.
            suffix (str, optional): Suffix to add to the dataset file name. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the generated dataset, consisting of:
                - alpha: A tensor of shape (N, 6) containing the alpha values used to generate the dataset.
                - theta: A tensor of shape (N, I, 6) containing the theta values used to generate the dataset.
                - x: A tensor of shape (N, I, J, 2, M) containing the generated sequences with noise.
        """
        M = self.bn['M']
        x_noise_level = self.bn['x_noise_level']

        # Sample N alpha values (for 6 parameters)
        alpha = self._p_alpha().sample((N, )).view(N, self.alpha_parameter_count)
        # Sample I theta values for each alpha (for 6 parameters)
        theta = torch.stack([self._p_theta_alpha(alpha_i).sample((I, )) for alpha_i in alpha]).view(N, I, 6)
        # Map the theta values to the correct range (we are using the module level variable `hard_constraints` here)
        theta = self.settings.map_thetas(theta, self.settings.constraints)
        # Solve the Lotka-Volterra model for each Theta
        x_true = self._p_x_theta(theta.view(N*I, 6)).view(N, I, 2, M)
        # Sample J noise values for each Theta
        noise_mean = torch.zeros_like(x_true)
        noise_std = (x_true * x_noise_level) + 0.00001
        noise_dist = dist.Normal(noise_mean, noise_std)
        x_noise = noise_dist.sample((J, )).view(N, I, J, 2, M)
        x = x_true.unsqueeze(dim=2).repeat_interleave(J, dim=2) + x_noise

        if logger and self.bn['save_data']:
            name = f"{suffix}"
            logger.save_data(name, (alpha, theta, x))

        return alpha, theta, x

    @staticmethod
    def group_up_data(alpha: torch.Tensor,
                      theta: torch.Tensor,
                      x: torch.Tensor,
                      flatten: bool = False):
        P = alpha.shape[-1]
        N, I, J, _, M = x.shape

        if flatten:
            alpha_shape = (N * I * J, P)
            theta_shape = (N * I * J, 6)
            x_shape = (N * I * J, 2, M)
        else:
            alpha_shape = (N, I, J, P)
            theta_shape = (N, I, J, 6)
            x_shape = (N, I, J, 2, M)
        alpha = alpha.repeat_interleave(I * J, dim=0).view(alpha_shape)
        theta = theta.repeat_interleave(J, dim=0).view(theta_shape) 
        x = x.view(x_shape)

        return alpha, theta, x
