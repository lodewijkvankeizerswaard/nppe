"""
This file contains the code to compute and visualise the posterior. We use functions from
`posterior_utils.py` to do this. 
"""

import torch
import torch.distributions as dist
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
import seaborn as sns

from distributions import ClippedGaussian
from posterior_utils import IntergrationDataset, PosteriorRatioGather, interpret_p_theta_alpha, line_plot, swarm_plot, kde_plot
from settings import ExperimentSettings, get_constraint_map
from utils import fig_to_array, Logger

def deep_set_model(ratios: torch.Tensor) -> torch.Tensor:
    """ A deepset model which assumes the input is flat. """
    # Currently this model is not aware of NIJ and K, and since 
    # we sum over N, I and J, and take the mean over K, this will not 
    # output the correct posterior ratio. It is scaled by a constant if I > 1 or J > 1.
    return ratios.exp().sum()

class Posterior:
    def __init__(self, X_test: torch.Tensor, settings: ExperimentSettings):
        self.settings = settings
        self.K = settings.posterior['K']
        # print(f"K: {self.K}")

        # First we need to construct the grid / samples over which we'll compute the posterior
        select_axis = settings.training['select_parameter_dims'] if settings.training['select_parameter_dims'] != 'all' else None
        self.axis_names, alpha_grid = interpret_p_theta_alpha(settings, select_axis=select_axis)
        self.P = len(self.axis_names)

        # Calculate pi(alpha)
        min_max_map = get_constraint_map((0, 1), select_axis)
        alpha_0 = torch.zeros((1, self.P))
        alpha_1 = torch.ones((1, self.P))
        alpha = torch.cat([alpha_0, alpha_1], dim=0)
        mapped_alpha = min_max_map(alpha)
        self.pi_alpha = 1 / torch.prod(mapped_alpha[1] - mapped_alpha[0])

        # Then we need to construct the dataset that will be used to integrate over the posterior space
        self.integration_dataset = IntergrationDataset(X_test, alpha_grid)
        self.integration_dataset_size = len(self.integration_dataset)
        integration_batch_size = settings.training['batch_size'] // self.K
        # print(f"integration_batch_size: {integration_batch_size}")
        self.integration_dataloader = torch.utils.data.DataLoader(self.integration_dataset, batch_size=integration_batch_size, shuffle=False)

        # We need to create an object that will gather the posterior ratios
        self.prg = PosteriorRatioGather(NIJ=X_test.shape[0],
                                        K=self.K,
                                        batch_size=integration_batch_size * self.K,
                                        res=settings.posterior['resolution'],
                                        P=self.P,
                                        deep_set_model=deep_set_model)
        
        self.p_t_a_model = partial(p_theta_alpha_model, settings=settings)

        # And we need to record the true values of the parameters for plotting
        self.true_values = [settings.bn['p_alpha_args'][g] if g in settings.bn['p_alpha_args'] else None for g in self.axis_names]


    @torch.no_grad()
    def compute_posterior(self, model: torch.nn.Module, save_data: bool = False, logger: Logger | None = None, verbose: bool = True) -> pd.DataFrame:
        """ Compute the posterior over alpha given the model and the dataset. """
        # Loop over the dataset and compute the posterior logratios
        total = int(np.ceil(self.integration_dataset_size / self.integration_dataloader.batch_size))
        for i, (x, a) in tqdm(enumerate(self.integration_dataloader), desc="Integrating posterior", total=total, disable=not verbose, leave=False):
            B = x.shape[0]

            x = x.repeat(self.K, 1, 1).to(model.device())

            # Sample K thetas for each alpha
            theta = self.p_t_a_model(a).sample((self.K,)).view(B*self.K, a.shape[-1]).to(model.device())

            # Map the theta values to the correct range
            theta_map = get_constraint_map((0, 1), self.settings.training['select_parameter_dims'])
            theta = theta_map(theta)

            ratios = model(x, theta).squeeze(-1).cpu()

            self.prg(ratios, i)
        
        # Get the output of the deepset model
        posterior = torch.log(self.prg.get_output()) - np.log(self.K) #+ torch.log(self.pi_alpha)

        posterior_df = pd.DataFrame(torch.cat([self.integration_dataset.alpha_grid, posterior.unsqueeze(1)], dim=1).cpu().detach().numpy(), 
                          columns=self.axis_names + ["logpost"])
        # self._map_posterior_to_bounds(posterior_df)

        if save_data and logger:
            logger.save_dataframe("posterior", posterior_df)


        return posterior_df
    
    def hyper_to_parameter(self, posterior: pd.DataFrame, grid: torch.Tensor) -> pd.DataFrame:
        # Get the posterior grid the grid
        # grid = self.integration_dataset.alpha_grid
        parameter_posterior = torch.empty((grid.shape[0]))

        # The last column is the 
        evaluation_points = torch.tensor(posterior.values)

        alpha_map = get_constraint_map((0, 1), self.settings.training['select_parameter_dims'])

        for ep in tqdm(evaluation_points, desc="Computing parameter posterior"):
            alpha, logprob = ep[:-1], ep[-1]
            alpha = alpha.unsqueeze(0)
            alpha = alpha_map(alpha)
            p_t_a_dist = self.p_t_a_model(alpha)
            p_t_a = p_t_a_dist.log_prob(grid)

            parameter_posterior += (p_t_a + logprob.item()).exp()

        parameter_posterior /= len(evaluation_points)

        parameter_posterior = torch.cat([grid, parameter_posterior.unsqueeze(1)], dim=1)

        return pd.DataFrame(parameter_posterior)

    @staticmethod
    def predict(df: pd.DataFrame, smooth: bool = False) -> torch.Tensor:
        prediction = torch.empty((len(df.columns[:-1])))
        

        for i, column in enumerate(df.columns[:-1]):
            # Get the mean log probability for each value of the parameter
            df_col = df.groupby(column).mean()
            if not smooth:
                column_prediction = df_col['logpost'].idxmax()
            else:
                # Calculate the resolution
                size = len(df.columns[:-1])
                resolution = torch.pow(torch.tensor([len(df)]), torch.tensor([1 / size])).long().item()
                # Bin the values and take the mean
                df_col['binned'] = pd.cut(df_col.index, bins=resolution, include_lowest=True)

                df_col = df_col.groupby('binned').mean()
                column_prediction = df_col['logpost'].idxmax().mid
            
            prediction[i] = float(column_prediction)

        return prediction

    @staticmethod    
    def evaluate(best_model_dict, test_set, settings, verbose: bool = None, return_posteriors: bool = False):
        metrics = {
            'l1': lambda x, y: torch.linalg.norm(x - y, ord=1),
            'l2': lambda x, y: torch.linalg.norm(x - y),
            'linf': lambda x, y: torch.linalg.norm(x - y, ord=float('inf')),
            'lninf': lambda x, y: torch.linalg.norm(x - y, ord=-float('inf'))
        }

        verbose = verbose if verbose is not None else settings.logging['verbose']
        results = defaultdict(list)
        posteriors = []

        select_axis = settings.training['select_parameter_dims'] if settings.training['select_parameter_dims'] != 'all' else [0, 1, 2, 3, 4, 5]
        for Ai, Di in zip(test_set[0][:settings.posterior['test_points']], test_set[2][:settings.posterior['test_points']]):
            Di = Di.unsqueeze(0)
            # Compute the posterior and calculate the distance to the true value
            posterior = Posterior(Di, settings)
            posterior_data = posterior.compute_posterior(best_model_dict['model'], verbose=verbose)
            smooth = not (settings.posterior['p_alpha'] == 'grid')
            prediction = posterior.predict(posterior_data, smooth=smooth)

            Ai_mapping = get_constraint_map((0, 1), select_axis)
            Ai = Ai[select_axis]
            Ai = Ai_mapping(Ai)

            for k, v in metrics.items():
                results[k].append(v(Ai, prediction))

                # Compute the best, worst and random distances to the true value (use cdist here)
                true_value_distances = [v(Ai, torch.tensor(posterior_data.iloc[[i]].values)[0][:-1]) for i in range(len(posterior_data))]
                results[f'{k}-best'].append(min(true_value_distances))
                results[f'{k}-worst'].append(max(true_value_distances))
                results[f'{k}-random'].append(true_value_distances[np.random.randint(0, len(true_value_distances))])



            posteriors.append(posterior_data)
            
        results = {k: torch.tensor(v) for k, v in results.items()}

        if return_posteriors:
            return results, posteriors
        return results
    
    def _map_posterior_to_bounds(self, posterior: pd.DataFrame) -> None:
        """ Map the posterior to the correct bounded space. """
        sorted_axes = ['alpha', 'beta', 'gamma', 'delta', 'x0', 'y0']
        for i, an in enumerate(self.axis_names):
            if 'mu' in an or 'lambda' in an:
                posterior.iloc[:, i] = posterior.iloc[:, i] * \
                                       (self.settings.constraints[sorted_axes[i] + '_upper_bound'] - \
                                        self.settings.constraints[sorted_axes[i] + '_lower_bound']) + \
                                        self.settings.constraints[sorted_axes[i] + '_lower_bound']
        
    @torch.no_grad()
    def plot_posterior(self, 
                       posterior: pd.DataFrame,
                       plot_types: list[str] = ['line'],
                       logger: Logger | None = None,
                       plot_std: bool = True,
                       true_values: dict[str, float] | None = None
                       ) -> tuple[list[matplotlib.figure.Figure], list[matplotlib.axis.Axis]] | None:
        """ Plot the posterior. """
        if not plot_types:
            return
        nr_plots = self.P
        nr_columns = 6
        nr_rows = int(np.ceil(nr_plots / nr_columns))
        figures = []
        axes = []

        if 'line' in plot_types:
            line_fig, line_axes = line_plot(posterior, nr_rows, nr_columns, plot_std=plot_std)
            self._add_true_values(line_axes, true_values)
            figures.append(line_fig)
            axes.append(line_axes)

            if logger:
                line_img = fig_to_array(line_fig)
                logger.log_image("post_line", line_img)

        if 'swarm' in plot_types:
            swarm_fig, swarm_axes = swarm_plot(posterior, nr_rows, nr_columns)
            self._add_true_values(swarm_axes, true_values)
            figures.append(swarm_fig)
            axes.append(swarm_axes)
            if logger:
                swarm_img = fig_to_array(swarm_fig)
                logger.log_image("post_swarm", swarm_img)

        if 'kde' in plot_types:
            kde_fig, kde_axes = kde_plot(posterior, nr_rows, nr_columns)
            self._add_true_values(kde_axes, true_values)
            figures.append(kde_fig)
            axes.append(kde_axes)
            if logger:
                kde_img = fig_to_array(kde_fig)
                logger.log_image("post_kde", kde_img)

    def pair_plot_posterior(self, 
                    posterior: pd.DataFrame,
                    logger: Logger | None = None,
                    plot_std: bool = True,
                    true_values: dict[str, float] | None = None,
                    parameter_posterior: pd.DataFrame | None = None,
                    parameter_grids: list[torch.Tensor] | None = None,
                    settings: ExperimentSettings  | None = None,
                    p_alpha: str = None,
                    ) -> tuple[list[matplotlib.figure.Figure], list[matplotlib.axis.Axis]] | None:
        
        size = len(posterior.columns[:-1])
        width = 2 * size

        resolution = torch.pow(torch.tensor([len(posterior)]), torch.tensor([1 / size])).long().item()

        fig, axes = plt.subplots(size, size, figsize=(width, width), squeeze=False)
        for i in range(size):
            for j in range(size):
                if i == j:
                    if p_alpha == 'grid':
                        sns.lineplot(data=posterior, x=posterior.columns[i], y='logpost', ax=axes[i, j], errorbar=None)

                    else:
                        sns.scatterplot(x=posterior[posterior.columns[i]], y=posterior['logpost'], ax=axes[i, j], marker='.', s=3)

                        df_col = posterior.groupby(posterior.columns[i]).mean()
                        df_col['binned'] = pd.cut(df_col.index, bins=resolution, include_lowest=True)
                        df_col['binned'] = df_col['binned'].apply(lambda x: x.mid)
                        df_col = df_col.groupby('binned').mean()
                        sns.lineplot(x=df_col.index, y=df_col['logpost'], ax=axes[i, j])
                    axes[i, j].set_xlabel("")
                    axes[i, j].set_ylabel("")
                elif i > j:
                    if parameter_posterior is not None:
                        df_parpost = parameter_posterior.groupby([i, j]).mean()
                        
                        x = np.unique(df_parpost.index.get_level_values(j).to_numpy())
                        y = np.unique(df_parpost.index.get_level_values(i).to_numpy())
                        parpost_size = len(np.unique(x))
                        z = df_parpost.iloc[:,-1].to_numpy()
                        axes[i, j].contourf(x, y, z.reshape((parpost_size, parpost_size)), cmap='Blues')

                    else:
                        axes[i, j].axis("off")
                else:
                    marker = '.' if p_alpha == 'uniform' else None
                    sns.scatterplot(data=posterior, x=posterior.columns[j], y=posterior.columns[i], hue='logpost', ax=axes[i, j], palette='Blues', marker=marker)
                    axes[i, j].legend().remove()
                    axes[i, j].set_xlabel("")
                    axes[i, j].set_ylabel("")

        fig.tight_layout()

        self._add_true_values_pair(axes, true_values)
        if logger:
            pair_img = fig_to_array(fig)
            logger.log_image("post_pair", pair_img)

        return fig, axes


    def _add_true_values(self, ax: matplotlib.axis.Axis, true_values: dict[str, float] | None = None) -> None:
        true_values = self.true_values if true_values is None else true_values
        for i, a in enumerate(ax):
            if true_values[i] is not None:
                a.axvline(true_values[i], color="red")

    def _add_true_values_pair(self, axes: matplotlib.axis.Axis, true_values: dict[str, float] | None = None) -> None:
        true_values = self.true_values if true_values is None else true_values
        for i, ax in enumerate(axes):
            for j, a in enumerate(ax):
                if i == j:
                    a.axvline(true_values[i], color="black", lw="1")
                elif i < j:
                    a.scatter(true_values[j], true_values[i], color="black", marker="x", )
                # if true_values[i] is not None:
            

def p_theta_alpha_model(alpha: torch.tensor, settings: ExperimentSettings) -> dist.Distribution:
    """ Returns a distribution with the given parameters. This is 
        a different function than the one in lotka_volterra.py because it reflects the modeling 
        choice and not the generative choice. """
    # alpha is in the constrained space, so we need to map it to [0, 1]
    mapping = get_constraint_map((0, 1), settings.training['select_parameter_dims'], inverse=True)
    alpha = mapping(alpha)
    if settings.posterior['p_theta_alpha'] == 'continuousbern':
        return dist.continuous_bernoulli.ContinuousBernoulli(probs=alpha)
    elif settings.posterior['p_theta_alpha'] == 'gaussian-means':
        mu = alpha
        sigma = torch.eye(alpha.shape[-1]).unsqueeze(0) * settings.posterior['p_theta_alpha_sigma']
        sigma = sigma.repeat(alpha.shape[0], 1, 1)
        d = dist.MultivariateNormal(mu, sigma)
        return d
        sigma = torch.eye(alpha.shape[-1]).unsqueeze(0) * settings.posterior['p_theta_alpha_sigma']
        return ClippedGaussian(mus=alpha, sigmas=sigma.repeat(alpha.shape[0], 1, 1))
    else:
        raise ValueError(f'Unknown p_theta_alpha: {settings.posterior["p_theta_alpha"]}')

if __name__ == "__main__":
    print("""Example usage of the posterior.py file.""")

    # Load the settings
    settings = ExperimentSettings.from_json_file("example_config.json")
    settings.posterior['resolution'] = 5
    settings.posterior['K'] = 30
    settings.training['batch_size'] = 1024

    # Create dummy data
    NIJ = 5
    channels = 1
    D = 3
    X_test = torch.arange(0, NIJ).repeat_interleave(D).view(NIJ, channels, D).type(torch.float32)
    print(X_test)

    # Create the posterior object and overwrite with interpretable parameters
    posterior = Posterior(X_test, settings)
    posterior.axis_names = ['alpha', 'beta', 'gamma']
    posterior.P = 3

    alpha_grid = torch.stack(torch.meshgrid([torch.linspace(0, 1, settings.posterior['resolution']) for _ in range(posterior.P)]), dim=-1).view(-1, posterior.P)
    posterior.integration_dataset = IntergrationDataset(X_test, alpha_grid)
    posterior.integration_dataset_size = len(posterior.integration_dataset)
    integration_batch_size = settings.training['batch_size'] // posterior.K
    posterior.integration_dataloader = torch.utils.data.DataLoader(posterior.integration_dataset, batch_size=integration_batch_size, shuffle=False)

    posterior.prg = PosteriorRatioGather(NIJ=X_test.shape[0],
                                    K=posterior.K,
                                    batch_size=integration_batch_size * posterior.K,
                                    res=settings.posterior['resolution'],
                                    P=posterior.P,
                                    deep_set_model=deep_set_model)

    # Create a dummy model
    class dummy_model(torch.nn.Module):
        def device(self):
            return "cpu"
        def forward(self, x, theta):
            return theta.mean(dim=2).squeeze(-1)
    
    # Compute the posterior
    posterior_data = posterior.compute_posterior(dummy_model(), map_thetas=False)
    posterior_plots = posterior.plot_posterior(posterior_data, plot_types=['pair'])

    import matplotlib.pyplot as plt
    plt.savefig("posterior.png")



