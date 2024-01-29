"""
This file contains the code to aggregate the results of multiple runs. By selecting runs from the 
`runs/` directory using the command line arguments, this script will aggregate the results and save
a distance metrics dataframe to the `results/` directory. This dataframe can then be used to plot
the results using the `distance_K.ipynb` notebook. 

This file is also used to generate singe-point and population-level posteriors.
"""

import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch.multiprocessing as mp

from settings import ExperimentSettings, get_constraint_map
from lotka_volterra import get_data, BNLotkaVolterra
from models import get_model
from posterior import Posterior
from posterior_utils import interpret_p_theta_alpha

from tqdm import tqdm

def get_runs() -> list[str]:
    """
    Get a list of file paths for the runs in `runs/uniform/gaussian-means`.

    Returns:
        A list of file paths for the runs.
    """
    base_path = os.path.join("runs", "uniform", "gaussian-means")
    runs = os.listdir(base_path)
    return [os.path.join(base_path, run) for run in runs]

def load_settings(run_path: str) -> ExperimentSettings:
    """
    Load experiment settings from a given run path.

    Args:
        run_path (str): The path to the run directory.

    Returns:
        ExperimentSettings: The loaded experiment settings.
    """
    config_path = os.path.join(run_path, "config.json")
    settings = ExperimentSettings.from_json_file(config_path)
    return settings

def load_run(run_path: str, settings: ExperimentSettings) -> tuple[torch.utils.data.Dataset, torch.nn.Module]:
    """
    Load the trained model and dataset for a specific run.

    Args:
        run_path (str): The path to the run directory.
        settings (ExperimentSettings): The experiment settings.

    Returns:
        tuple[torch.utils.data.Dataset, torch.nn.Module]: The loaded dataset and model.
    """
    model_path = os.path.join(run_path, "models", "model.pt")
    model = get_model(settings)
    model.load_state_dict(torch.load(model_path))
    data = get_data(settings)
    return data, model

def select_runs(agg_conf: dict) -> list[tuple[str, ExperimentSettings]]:
    """
    Selects runs based on the given aggregation configuration.

    Args:
        agg_conf (dict): The aggregation configuration containing the criteria for selecting runs.

    Returns:
        list[tuple[str, ExperimentSettings]]: A list of tuples containing the run name and its corresponding settings.
    """
    runs = get_runs()
    run_settings = [load_settings(run) for run in runs]
    selected_runs = [(run, settings) for run, settings in zip(runs, run_settings) 
        if (settings.logging['runid'] == agg_conf['runid'] or agg_conf['runid'] is None) \
            and (settings.training['lr'] == agg_conf['learning_rate'] or agg_conf['learning_rate'] is None) \
            and (settings.training['select_parameter_dims'] == agg_conf['select_parameter_dims'] or agg_conf['select_parameter_dims'] is None) \
            and (settings.training['hidden_size'] == agg_conf['hidden_size'] or agg_conf['hidden_size'] is None) \
            and (agg_conf['run'] in settings.name)
            and (settings.training['param_dropout'] == agg_conf['param_dropout'] or agg_conf['param_dropout'] is None)
    ]
    return selected_runs

def get_save_dir(agg_conf: dict) -> str:
    """
    Constructs the save directory path based on the given aggregation configuration.

    Args:
        agg_conf (dict): The aggregation configuration containing the necessary parameters.

    Returns:
        str: The constructed save directory path.
    """
    save_dir = os.path.join("results",
                            f"{agg_conf['runid']}",
                            f"{agg_conf['param_dropout']}",
                            f"{agg_conf['learning_rate']}",
                            f"{agg_conf['select_parameter_dims']}",
                            f"{agg_conf['hidden_size']}",
                            f"{agg_conf['p_alpha']}",
                            f"{agg_conf['test_points']}")
    return save_dir

def aggregate_runs(selected_runs: list, agg_conf: dict, save_dir: str) -> tuple[pd.DataFrame, str]:
    filename = os.path.join(save_dir, f"pta_sigma-K-{agg_conf['test_points']}-{agg_conf['resolution']}.csv")

    # Check if this aggregation has already been done
    if os.path.exists(filename):
        return pd.read_csv(filename), filename

    # Prepare results dataframe and save directory
    os.makedirs(save_dir, exist_ok=True)
    pta_sigma_K = pd.DataFrame(columns=
                               ["runid", "name", "seed", "lr", "select_parameter_dims", "p_theta_alpha_sigma", "K", "hidden_size", "x_noise_level", "param_dropout",
                                "l2", "l2-best", "l2-worst", "l2-random", "l2-std", "l2-best-std", "l2-worst-std", "l2-random-std",
                                "linf", "linf-best", "linf-worst", "linf-random", "linf-std", "linf-best-std", "linf-worst-std", "linf-random-std",
                                "lninf", "lninf-best", "lninf-worst", "lninf-random", "lninf-std", "lninf-best-std", "lninf-worst-std", "lninf-random-std",
                                "l1", "l1-best", "l1-worst", "l1-random", "l1-std", "l1-best-std", "l1-worst-std", "l1-random-std"]
                              )

    for run, settings in tqdm(selected_runs, desc="Evaluating runs"):
        # Create aliases for settings and ensure posterior settings are correct
        rid = settings.logging['runid']
        lr = settings.training['lr']
        spds = settings.training['select_parameter_dims']
        hs = settings.training['hidden_size']
        seed = settings.training['seed']
        x_noise = settings.bn['x_noise_level']
        settings.posterior['p_alpha'] = agg_conf['p_alpha']
        settings.posterior['resolution'] = agg_conf['resolution']
        settings.posterior['test_points'] = agg_conf['test_points']
        param_dropout = settings.training['param_dropout']

        # Load data and model
        (_, _,test_set), model = load_run(run, settings)

        # Move model to device
        model.to(settings.device)
        model.eval()
        # model.to(torch.device('cpu'))
        best_model_dict = {'model': model}

        for pta_sigma in tqdm(agg_conf['pta_sigmas'], desc="Evaluating pta_sigma", leave=False, position=1) :
            for k in tqdm(agg_conf['ks'], desc="Evaluating K", leave=False, position=2):
                settings.posterior['p_theta_alpha_sigma'] = pta_sigma
                results = Posterior.evaluate(best_model_dict, test_set, settings, verbose=True)
                info = [rid, settings.name, seed, lr, str(spds), pta_sigma, k, hs, x_noise, param_dropout]

                metrics = [func(results[f"{metric}{sort}"]).item() 
                                            for metric in ["l2", "linf", "lninf", "l1"] 
                                            for func in [torch.mean, torch.std]
                                            for sort in ["", "-best", "-worst", "-random"]]
                pta_sigma_K.loc[len(pta_sigma_K)] = info + metrics

    pta_sigma_K.to_csv(filename)
    return pta_sigma_K, filename

def plot_single_posterior(run: str, 
                          settings: ExperimentSettings, 
                          save_dir: str, 
                          pta_sigma: float, 
                          k: int, Di: int, 
                          p_alpha: str = "grid", 
                          resolution: int = 5, 
                          parameter_resolution: int = 15,
                          save_id: int = None,
                          include_prediction: bool = False):
    settings = load_settings(run)
    (_, _, test_set), model = load_run(run, settings)
    model.to(settings.device)

    settings.posterior['p_theta_alpha_sigma'] = pta_sigma
    settings.posterior['K'] = k
    settings.posterior['p_alpha'] = p_alpha
    settings.posterior['resolution'] = resolution

    D = test_set[2][Di].unsqueeze(0)
    alpha = test_set[0][Di][settings.training['select_parameter_dims']]
    alpha = get_constraint_map((0, 1), settings.training['select_parameter_dims'])(alpha)

    # Calculate posterior
    posterior = Posterior(D, settings)
    posterior_data = posterior.compute_posterior(model, verbose=True)
    
    # Calculate parameter posterior
    parameter_posterior_settings = {
        'resolution': parameter_resolution, 
        'p_alpha': "grid", 
        'p_theta_alpha': "gaussian-means",  # This has no influence on the grid but is read by `interpret_p_theta_alpha`
    }
    par_settings = ExperimentSettings({}, {'device': "", 'select_parameter_dims': settings.training['select_parameter_dims']}, parameter_posterior_settings , {}, "")
    _, grid_list, grids = interpret_p_theta_alpha(par_settings, select_axis=settings.training['select_parameter_dims'], return_grids=True)
    parameter_posterior = posterior.hyper_to_parameter(posterior_data, grid=grid_list)

    # Plot posterior
    for parampost in [parameter_posterior]:
        fig, ax = posterior.pair_plot_posterior(posterior_data, true_values=alpha, p_alpha=settings.posterior['p_alpha'], parameter_posterior=parampost)
        if include_prediction:
            smooth = not (settings.posterior['p_alpha'] == 'grid')
            prediction = posterior.predict(posterior_data, smooth=smooth)
            for i, p in enumerate(prediction):
                ax[i, i].axvline(x=p, color='red', linestyle='--')

        plt.tight_layout()

        # Save figure
        img_savedir = save_dir.replace("results", "img")
        img_filename = f"posterior-{Di}-{parampost is not None}.png" if save_id == None else f"posterior-{save_id}-{parampost is not None}.png"
        img_filename = os.path.join(img_savedir, img_filename)
        os.makedirs(os.path.dirname(img_filename), exist_ok=True)
        plt.savefig(img_filename, bbox_inches='tight', dpi=300)


def population_test(run: str, 
                    settings: ExperimentSettings, 
                    true_values: torch.Tensor,
                    save_dir: str, 
                    pta_sigma: float, 
                    k: int, 
                    p_alpha: str = "grid", 
                    resolution: int = 5,
                    parameter_resolution: int = 15,
                    save_id: int = None
                ):

    # Generate a test set with specific population level parameters
    train = {'nr_threads': 16, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    bn = {
        'p_alpha': 'gaussian',
        'p_alpha_args': {'mu_alpha_0': true_values[0], 
                        'mu_alpha_1': true_values[1],
                        'mu_alpha_2': true_values[2],
                        'mu_alpha_3': true_values[3],
                        'mu_alpha_4': 0.5,
                        'mu_alpha_5': 0.5,
                        'sigma_alpha_0': 0.1,
                        'sigma_alpha_1': 0.3,
                        'sigma_alpha_2': 0.1,
                        'sigma_alpha_3': 0.3,
                        'sigma_alpha_4': 0.5,
                        'sigma_alpha_5': 0.5,},
        'p_theta_alpha': 'gaussian-means',
        'p_theta_alpha_args': {},
        'M': 100,
        'dt': 0.1,
        'x_noise_level': 0.01,
    }
    gen_settings = ExperimentSettings(bn, train, {}, {}, name="runs/population-test")
    BN = BNLotkaVolterra(gen_settings)
    data = BN.generate_dataset(N=100, I=1, J=1)
    data = BN.group_up_data(*data, flatten=True)
    _, _, D = data

    # Load model
    _, model = load_run(run, settings)
    model.to(settings.device)

    # Compute posterior
    settings.posterior['p_theta_alpha_sigma'] = pta_sigma
    settings.posterior['K'] = k
    settings.posterior['p_alpha'] = p_alpha
    settings.posterior['resolution'] = resolution

    posterior = Posterior(D, settings)
    posterior_data = posterior.compute_posterior(model, verbose=True)
    posterior_data.to_csv("results/population_posterior.csv")

    # Calculate parameter posterior
    parameter_posterior_settings = {
        'resolution': parameter_resolution, 
        'p_alpha': "grid", 
        'p_theta_alpha': "gaussian-means",  # This has no influence on the grid but is read by `interpret_p_theta_alpha`
    }
    par_settings = ExperimentSettings({}, {'device': "", 'select_parameter_dims': settings.training['select_parameter_dims']}, parameter_posterior_settings , {}, "")
    _, grid_list, grids = interpret_p_theta_alpha(par_settings, select_axis=settings.training['select_parameter_dims'], return_grids=True)
    parameter_posterior = posterior.hyper_to_parameter(posterior_data, grid=grid_list)

    alpha_map = get_constraint_map((0, 1), settings.training['select_parameter_dims'])
    true_values = alpha_map(true_values)

    for parampost in [parameter_posterior]:
        fig, ax = posterior.pair_plot_posterior(posterior_data, true_values=true_values, p_alpha=settings.posterior['p_alpha'], parameter_posterior=parampost)

        # Save figure
        img_save_dir = save_dir.replace("results", "img")
        img_filename = os.path.join(img_save_dir, f"population-posterior-{parampost is not None}-{save_id}.png")
        os.makedirs(os.path.dirname(img_filename), exist_ok=True)
        plt.savefig(img_filename, bbox_inches='tight', dpi=300)
    


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", type=str, default=None)
    parser.add_argument("--param_dropout", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--p_alpha", type=str, default="grid", choices=["grid", "uniform"])
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--parameter_resolution", type=int, default=15)
    parser.add_argument("--test_points", type=int, default=10)
    parser.add_argument("--compute_distance_metrics", action="store_true")
    parser.add_argument("--plot_sigma_K", action="store_true")
    parser.add_argument("--posteriors_sigma", type=float, default=0.1)
    parser.add_argument("--single_posteriors", type=int, default=0)
    parser.add_argument("--population_posteriors", type=int, default=0)
    parser.add_argument("--run", type=str, default="", help="Run to obtain plots for a single run.")
    args = parser.parse_args()
    print(args)

    agg_conf = {
        'runid': args.runid,
        'param_dropout': args.param_dropout,
        'learning_rate': args.learning_rate,
        'select_parameter_dims': [0, 1, 2, 3],
        'hidden_size': args.hidden_size,
        'p_alpha': args.p_alpha,
        'pta_sigmas': [0.03, 0.05, 0.1],
        'ks': [10, 32, 100, 316, 1000],
        'test_points': args.test_points,
        'resolution': args.resolution,
        'run': args.run,
    }
    selected_runs = select_runs(agg_conf)
    print(f"Selected {len(selected_runs)} runs")

    save_dir = get_save_dir(agg_conf)
    if args.compute_distance_metrics:
        df, filename = aggregate_runs(selected_runs, agg_conf, save_dir)

    if args.compute_distance_metrics and args.plot_sigma_K:
        print("Please use the `distance_K.ipynb` notebook to plot the results.")

    Di = torch.randperm(1000)[:args.single_posteriors]
    for i, d in enumerate(Di):
        sr = i % len(selected_runs)
        run = selected_runs[sr][0]
        settings = selected_runs[sr][1]

        plot_single_posterior(run, settings, save_dir, args.posteriors_sigma, 1000, d, agg_conf['p_alpha'], agg_conf['resolution'], args.parameter_resolution, i)

    true_value_dist = torch.distributions.Uniform(0, 1)
    for i in range(args.population_posteriors):
        true_values = true_value_dist.sample((len(agg_conf['select_parameter_dims']),))
        sr = i % len(selected_runs)
        run = selected_runs[sr][0]
        settings = selected_runs[sr][1]
        population_test(run, settings, true_values, save_dir, args.posteriors_sigma, 1000, agg_conf['p_alpha'], agg_conf['resolution'], args.parameter_resolution, i)



