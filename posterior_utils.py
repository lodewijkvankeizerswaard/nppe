"""
This file contains an IntegratioDataset, which is used to iterate over the entire
hyper-data space. It also contains a PosteriorRatioGather, which is used to gather
the posterior ratio for each batch of data, apply a deepset model to the appropriate
batch of data, and then saves the posterior over the grid.

Example usage can be found in the `__main__` portion of this file.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from settings import get_constraint_map

class IntergrationDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for integration. It takes a grid of alpha values and a test set
    and iterates over all combinations of alpha and test data, effectively containing the 
    entire integration space. This is coded to prevent memory issues, by only storing the
    data once, and computing the indices on the fly.

    Args:
        X_test (torch.Tensor): The test data.
        alpha_grid (torch.Tensor): The grid of alpha values.
        len_grid (int | None, optional): The length of the grid. Defaults to None.
    """
    def __init__(self, X_test: torch.Tensor, alpha_grid: torch.Tensor, len_grid: int | None = None):
        self.X_test = X_test
        self.alpha_grid = alpha_grid
        self.len_X_test = X_test.shape[0]
        self.len_grid = len_grid if len_grid else len(alpha_grid)
    
    def __len__(self):
        return self.len_grid * self.len_X_test
    
    def __getitem__(self, idx):
            """
            Get the item at the given index or slice. If a slice is given, the indices are
            converted to a numpy array. The output can be spread over multiple grid points.
            This is something to keep in mind when using this dataset.

            Parameters:
            idx (int or slice): The index or slice to retrieve.

            Returns:
            tuple: A tuple containing the X_test value and the alpha_grid value.
            """
            if isinstance(idx, slice):
                idx = np.array(list(range(idx.stop)[idx]))
            X_idx = idx % self.len_X_test
            grid_idx = idx // self.len_X_test
            return self.X_test[X_idx], self.alpha_grid[grid_idx]
    
def interpret_p_theta_alpha(settings, select_axis: list[int] | None = None, return_grids: bool = False):
    """ Returns a grid over the parameter space of `p_theta_alpha`, and the corresponding names. 
        Since the true form of `p_theta_alpha` is unknown, we use a choosen distribution to
        model it. """
    res = settings.posterior['resolution']
    indexing = 'ij'
    if settings.posterior['p_theta_alpha'] == 'continuousbern':
        # In this case we have 6 lambda parameters
        # axis_names = ['lambda_alpha', 'lambda_beta', 'lambda_gamma', 'lambda_delta', 'lambda_x0', 'lambda_y0']
        names = [f'lambda_alpha_{i}' for i in range(6)]
        axis_names = [names[i] for i in select_axis] if select_axis else names
        if settings.posterior['p_alpha'] == 'grid':
            spaces = [torch.linspace(0, 1, res) for _ in range(6)]
            selected_spaces = [spaces[i] for i in select_axis] if select_axis else spaces
            grids = torch.meshgrid(*selected_spaces, indexing=indexing)
        elif settings.posterior['p_alpha'] == 'uniform':
            grid_list = torch.rand((res**len(select_axis), len(select_axis))) if select_axis else torch.rand((res**6, 6))

    elif settings.posterior['p_theta_alpha'] == 'gaussian':
        # In this case we have 6 mu parameters and 6 sigma parameters
        # axis_names = ['mu_alpha', 'mu_beta', 'mu_gamma', 'mu_delta', 'mu_x0', 'mu_y0',
                        # 'sigma_alpha', 'sigma_beta', 'sigma_gamma', 'sigma_delta', 'sigma_x0', 'sigma_y0']
        mean_axis_names = [f'mu_alpha_{i}' for i in range(6)]
        std_axis_names = [f'sigma_alpha_{i}' for i in range(6)]
        axis_names = [mean_axis_names[i] for i in select_axis] + [std_axis_names[i] for i in select_axis] if select_axis else mean_axis_names
        axis_names += [std_axis_names[i] for i in select_axis] if select_axis else std_axis_names

        if settings.posterior['p_alpha'] == 'grid':
            mean_spaces = [torch.linspace(0, 1, res) for _ in range(6)]
            std_spaces = [torch.linspace(0.001, 1, res) for _ in range(6)]
            selected_mean_spaces = [mean_spaces[i] for i in select_axis] if select_axis else mean_spaces
            selected_std_spaces = [std_spaces[i] for i in select_axis] if select_axis else std_spaces

            selected_spaces = selected_mean_spaces + selected_std_spaces

            grids = torch.meshgrid(selected_spaces, indexing=indexing)
        elif settings.posterior['p_alpha'] == 'uniform':
            grid_list = torch.rand((res**len(select_axis), len(select_axis))) if select_axis else torch.rand((res**12, 12))

    elif settings.posterior['p_theta_alpha'] == 'gaussian-means':
        # In this case we have 6 mu parameters
        # axis_names = ['mu_alpha', 'mu_beta', 'mu_gamma', 'mu_delta', 'mu_x0', 'mu_y0']
        names = [f'mu_alpha_{i}' for i in range(6)]
        axis_names = [names[i] for i in select_axis] if select_axis else names
        if settings.posterior['p_alpha'] == 'grid':
            spaces = [torch.linspace(0, 1, res) for _ in range(6)]
            selected_spaces = [spaces[i] for i in select_axis] if select_axis else spaces
            grids = torch.meshgrid(*selected_spaces, indexing=indexing)
        elif settings.posterior['p_alpha'] == 'uniform':
            grid_list = torch.rand((res**len(select_axis), len(select_axis))) if select_axis else torch.rand((res**6, 6))
        # spaces = [torch.linspace(0, 1, res) for _ in range(6)]
        # selected_spaces = [spaces[i] for i in select_axis] if select_axis else spaces
        # grids = torch.meshgrid(*selected_spaces, indexing=indexing)
    else:
        raise ValueError(f'Unknown p_theta_alpha: {settings.posterior["p_theta_alpha"]}')

    # The grid_list is a list of all possible values for the parameters, and
    # will be of shape (res**P, P) where P is the amount of parameters
    if settings.posterior['p_alpha'] == 'grid':
        grid_list = torch.stack(grids, dim=-1).flatten(start_dim=0, end_dim=-2)

    # The names of the axis are in the same order as the final dimension of the grid_list
        
    # Map the grid_list onto the correct ranges
    mapping = get_constraint_map((0, 1), settings.training['select_parameter_dims'])

    # Apply the mapping to the grid_list
    grid_list = mapping(grid_list)

    if return_grids and settings.posterior['p_alpha'] == 'grid':
        return axis_names, grid_list, grids
    elif return_grids and settings.posterior['p_alpha'] == 'uniform':
        return axis_names, grid_list, None
    return axis_names, grid_list

class PosteriorRatioGather():
    def __init__(self, NIJ: int, K: int, batch_size: int, res: int, P: int, deep_set_model: callable) -> None:
        """ A class to gather posterior ratio for each batch of data, apply a deepset model
            to the appropriate batch of data, and then saves the posterior over the grid. 
            The general idea is that we want to call the DeepSet model for every coordinate,
            but we want to do this in batches. Lets say we have 4 coordinates, 
            NIJK = 4 and we fill the cache in batches of 3. After the first batch we have:

            (x0, x1) | NIJK_0, NIJK_1, NIJK_2, NIJK_3
            -----------------------------------------
            (0 , 0)  |  x    , x     , x     , _
            (0 , 1)  |  _    , _     , _     , _
            (1 , 0)  |  _    , _     , _     , _
            (1 , 1)  |  _    , _     , _     , _

            where `x` indicates a filled coordinate, and `_` indicates an empty coordinate.
            After the second batch we have a filled cache for the first coordinate, and we
            can apply the deepset model to it. We then get:

            (x0, x1) | NIJK_0, NIJK_1, NIJK_2, NIJK_3
            -----------------------------------------
            (0 , 0)  |  y
            (0 , 1)  |  x    , x     , _     , _
            (1 , 0)  |  _    , _     , _     , _
            (1 , 1)  |  _    , _     , _     , _

            where `y` indicates the output of the deepset model. We then repeat this process
            until all coordinates are filled. Since we take the batch_idx as input, we can
            fill the cache asynchroniously (i.e. using multiple processes).
        """
        self.NIJ = NIJ
        self.K = K
        self.batch_size = batch_size
        self.res = res
        self.P = P
        self.deep_set_model = deep_set_model

        self.nr_coordinate_points = res ** P
        # print(f"{self.nr_coordinate_points = }")
        self.ratios_per_coordinate = NIJ * K
        # print(f"{self.ratios_per_coordinate = }")

        cache_element = lambda : torch.zeros((self.ratios_per_coordinate,))

        self.cache = defaultdict(cache_element)
        self.count = {i: 0 for i in range(self.nr_coordinate_points)}

    def _apply_deep_set_model(self, coordinate_index: int) -> torch.Tensor:
        """ Applies the deepset model to the filled data in the cache. It replaces the
            caches ratios with the ouput of the deepset model, preventing enormous memory
            usage. """
        if self.count[coordinate_index] != self.ratios_per_coordinate:
            raise ValueError(f"Coordinate {coordinate_index} is not filled.")
        data = self.cache[coordinate_index]
        deep_set_ratio = self.deep_set_model(data)
        self.cache[coordinate_index] = deep_set_ratio

    def __call__(self, ratios_batch: torch.Tensor, batch_idx: torch.Tensor) -> None:
        # Construct the index in the NIJ * K * res**P space (i.e. the flattened set of ratios)
        global_ratios_index_range = torch.linspace(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1) - 1, self.batch_size, dtype=torch.long)
        # Construct the index in the coordinate space i.e. res**P (we have K * NIJ ratios per coordinate)
        coordinate_index_range = global_ratios_index_range // self.ratios_per_coordinate

        # If we are at the end of the entire data set, we need to remove all indices that are larger than NIJ * K * res**P
        if ratios_batch.shape[0] < self.batch_size:
            global_ratios_index_range = global_ratios_index_range[ : ratios_batch.shape[0]]
            coordinate_index_range = coordinate_index_range[ : ratios_batch.shape[0]]

        # Calculate the internal coordinate index for the batch in NIJ * K
        internal_coordinate_index = global_ratios_index_range % self.ratios_per_coordinate

        # Split the ratios_batch based on coordinate_index_range
        unique_coordinates, coordinate_chunk_sizes = torch.unique_consecutive(coordinate_index_range, return_counts=True)

        # Get the chunks of ratios_batch that will be split up to be placed in the cache
        ratio_chunks = torch.split(ratios_batch, coordinate_chunk_sizes.tolist())

        # Get the internal indices off for each chunk so they get stored in the appropriate place in the cache
        ratio_idx_chunks = torch.split(internal_coordinate_index, coordinate_chunk_sizes.tolist())

        # For each unique coordinate we place the ratios in their internal idx in the cache
        for unique_idx, (coordinate, internal_idx) in enumerate(zip(unique_coordinates, ratio_idx_chunks)):
            self.cache[coordinate.item()][internal_idx] = ratio_chunks[unique_idx]
            self.count[coordinate.item()] += len(internal_idx)

        # Check if the coordinate is filled
        for coordinate in unique_coordinates:
            if self.count[coordinate.item()] == self.ratios_per_coordinate:
                self._apply_deep_set_model(coordinate.item())

    def get_output(self) -> torch.Tensor:
        """ Returns the output of the deepset model. """
        # Check if all coordinates are filled
        for coordinate in self.count:
            if self.count[coordinate] != self.ratios_per_coordinate:
                raise ValueError(f"Coordinate {coordinate} is not filled. If this gets triggered it is most likely that\
                                 the data set used is not of the expected size.")
            
        # Get the output
        return torch.stack([self.cache[coordinate] for coordinate in range(self.nr_coordinate_points)])
    
    def __str__(self) -> str:
        s = f"PosteriorRatioGather(NIJ={self.NIJ}, K={self.K}, batch_size={self.batch_size}, res={self.res}, P={self.P})\n"
        # s += f"-------------------\n"
        # for i, v in self.cache.items():
        #     s += f"{i}: {v}\n"

        return s

def line_plot(posterior: pd.DataFrame, 
              nr_rows: int = 1,
              nr_columns: int = 6,
              width: int = 20,
              plot_std: bool = True,
            ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """ Plot the posterior as a line plot. """
    if isinstance(posterior, pd.DataFrame):
        posterior = [posterior]
    axis_names = posterior[0].columns[:-1]
    height = 4 * nr_rows

    line_fig, line_axes = plt.subplots(nr_rows, nr_columns, figsize=(width, height), squeeze=False)
    line_axes = line_axes.ravel()
    for i, v in enumerate(axis_names):
        for p in posterior:
            sns.lineplot(x=v, y="logpost", data=p, ax=line_axes[i], errorbar='sd' if plot_std else None, color='black')
        line_axes[i].set(xlabel=v)
        if i > 0:
            line_axes[i].set(ylabel="")
    line_fig.subplots_adjust(bottom=0.18, left=0.06)

    line_fig.tight_layout()
    return line_fig, line_axes

def swarm_plot(posterior: pd.DataFrame, 
              nr_rows: int = 1,
              nr_columns: int = 6,
              width: int = 20,
            ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """ Plot the posterior as a line plot. """
    axis_names = posterior.columns[:-1]
    height = 4 * nr_rows

    swarm_fig, swarm_axes = plt.subplots(nr_rows, nr_columns, figsize=(width, height), squeeze=False)
    swarm_axes = swarm_axes.ravel()
    for i, v in enumerate(axis_names):
        sns.swarmplot(x=v, y="logpost", data=posterior, ax=swarm_axes[i], size=0.3, native_scale=True)
        swarm_axes[i].set(xlabel=v)
        if i > 0:
            swarm_axes[i].set(ylabel="")
    swarm_fig.subplots_adjust(bottom=0.18, left=0.06)

    swarm_fig.tight_layout()
    return swarm_fig, swarm_axes

def kde_plot(posterior: pd.DataFrame,
                nr_rows: int = 1,
                nr_columns: int = 6,
                width: int = 20,
                ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """ Plot the posterior as a line plot. """
    axis_names = posterior.columns[:-1]
    height = 4 * nr_rows

    kde_fig, kde_axes = plt.subplots(nr_rows, nr_columns, figsize=(width, height), squeeze=False)
    kde_axes = kde_axes.ravel()
    for i, v in enumerate(axis_names):
        sns.kdeplot(x=v, y="logpost", data=posterior, ax=kde_axes[i], fill=True, legend=True)
        kde_axes[i].set(xlabel=v)
        if i > 0:
            kde_axes[i].set(ylabel="")
    kde_fig.subplots_adjust(bottom=0.18, left=0.06)

    kde_fig.tight_layout()
    return kde_fig, kde_axes

if __name__ == "__main__":
    print(""" An example usage of the IntergrationDataset. """)
    res = 3
    P = 2
    NIJ = 3
    X_test = torch.arange(NIJ).unsqueeze(-1)

    grids = torch.meshgrid([torch.linspace(0, 1, res) for _ in range(P)], indexing='xy')
    alpha_grid = torch.stack(grids, dim=-1).flatten(start_dim=0, end_dim=-2)

    integration_dataset = IntergrationDataset(X_test, alpha_grid)
    # To see the spreading of the data over multple grid points, set `batch_size` to 4
    integration_dataloader = torch.utils.data.DataLoader(integration_dataset, batch_size=4, shuffle=False)

    for x, a in integration_dataloader:
        # print(x, a)
        print(f"{x.shape = } {a.shape = }")

    print(f"{ len(integration_dataset) } == { (res ** P) * X_test.shape[0] }")

    print(""" An example usage of the PosteriorRatioGather. """)
    NIJ = 100
    res = 3
    P = 2
    K = 2
    batch_size = 12
    deep_set_model = lambda x: x.mean(dim=0)

    # X_test.shape = (N*I*J, 2, M), but here we'll use (N*I*J, 1)
    X_test = torch.arange(0, NIJ).unsqueeze(1).float()

    grids = torch.meshgrid(torch.linspace(0, 1, res),  # lambda of alpha
                            torch.linspace(0, 1, res),  # lambda of beta
                            indexing="ij")
        
    # Construct a grid of all possible values for the parameters
    alpha_grid = torch.stack(grids, dim=-1).flatten(start_dim=0, end_dim=-2)
    prk_count = alpha_grid.shape[0]
    thr_count = res ** P

    print(f"prk_count: {prk_count} == {thr_count} :thr_count")
    print(f"alpha_grid.shape: {alpha_grid.shape}")


    pid = IntergrationDataset(X_test, alpha_grid, len_grid=prk_count)

    posterior_batch_size = batch_size // K
    grid_dataloader = torch.utils.data.DataLoader(pid, batch_size=posterior_batch_size, shuffle=False)

    prg = PosteriorRatioGather(NIJ=NIJ,
                                 K=K,
                                 batch_size=grid_dataloader.batch_size * K,
                                 res=res,
                                 P=P,
                                 deep_set_model=deep_set_model)
    
    for i, (x, a) in enumerate(grid_dataloader):
        B = x.shape[0]

        x = x.repeat(K, 1)
        theta = a.repeat(K, 1)

        ratios = (theta * x).sum(dim=-1)
        prg(ratios, i)

    print(prg.get_output())