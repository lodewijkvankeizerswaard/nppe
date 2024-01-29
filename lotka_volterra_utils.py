"""
This file contains a parrelised implementation for generating the Lotka-Volterra dataset. 
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
import scipy.integrate as integrate
from tqdm import tqdm

from utils import fig_to_array

def solve_simulation(theta: torch.Tensor, M: int, dt: float = 0.1) -> torch.Tensor:
    """Simulate Lotka-Volterra model with given parameters and return the result."""
    # Check the input
    assert theta.shape == (6,)
    assert (theta > 0).all()

    # Simulate
    alpha, beta, gamma, delta, x0, y0 = theta.numpy()
    def lv_ivp(t, y: np.ndarray) -> np.ndarray:
        x, y = y
        return [alpha * x - beta * x * y, -gamma * y + delta * x * y]
    D = integrate.solve_ivp(lv_ivp, (0, M), [x0, y0], t_eval=np.arange(0, M * dt, dt), vectorized=True).y

    # Check the output
    assert D.shape == (2, M)
    return torch.from_numpy(D).float()

def solve_simulation_batch(thetas: torch.Tensor, M: int, dt: float = 0.1, processes = 1) -> torch.Tensor:
    """Simulate Lotka-Volterra model with given parameters and return the result."""
    # Check the input
    assert thetas.shape[1] == 6
    assert (thetas > 0).all()

    desc = "Generating LV"

    if processes == 1:
        # Simulate
        D = torch.stack([solve_simulation(theta, M, dt) for theta in tqdm(thetas, desc=desc)])

    else:
        # # Simulate pytorch==1.13
        pool = mp.Pool(processes)
        poolfunc = partial(solve_simulation, M=M, dt=dt)
        D = torch.stack(pool.map(poolfunc, thetas))
        pool.close()
        pool.join()

    # Check the output
    assert D.shape == (thetas.shape[0], 2, M)
    return D

def load_debug_dataset(name: str):
    file_name = f"data/debug/{name}.pt"
    dataset = torch.load(file_name)
    return dataset
