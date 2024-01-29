# DeepSet Hyper Posterior Estimator

This repository contains the code for Part 2 of the master thesis: ``Population-Level Density Estimation Using Normalizing Flows and Hyper Posteriors". Link will be added as soon as it is available online.

## Installation
To install the environment run:
```
conda env create -f environment.yml
```
To use the wandb logger, also install wandb:
```
pip install wandb==0.13.5
```

## Usage
To train the DHPE model with dropout on the Lotka-Volterra dataset run:
```
python main.py --bn.p_theta_alpha gaussian-means --bn.x_noise_level 0.01 --bn.train_N 50000 --bn.val_N 5000 --bn.test_N 1000 --training.model cnn --training.hidden_size 128 --training.hidden_layers_data 0 --training.hidden_layers_param 5 --training.hidden_layers_out 1 --training.lr 0.001 --training.clip_grad 0.1 --training.epochs 100 --training.nr_threads 8 --training.select_parameter_dims "[0, 1, 2, 3]" --training.batch_size 2048 --training.param_dropout 0.1 --training.milestones 50 
```
or use the command
```
python main.py --from_json example_config.json
```
to obtain a model quickly (though this does not perform well).

However, to reproduce the results from the paper, we recommend to use the `batch_run.py` script to submit multiple runs to a cluster. For example, to train the models for the experiments in the paper run:
```
python batch_run.py --add_seeds --add_logger --add_name --dry runfiles/experiments_runfile.md 
```
without `--dry`.

The performance metrics and example posteriors can be computed and plotted with:
```
python results.py --compute_posterior --single_posteriors 5 --population_posteriors 1
```
If there are multiple runs done, this script support selecting runs based on their arguments. 

## Files
This repository contains the following files:
- `batch_run.py`: A script to submit multiple experiment runs to a cluster (see `runfiles/experimets_runfile.md`).
- `main.py`: The main file to run the experiments.
- `train.py`: The training procedures and ratio dataset construction.
- `posterior.py`: The posterior estimator and Deepset model.
- `posterior_utils.py`: Helper functions and classes to compute the posterior.
- `models.py`: Ratio Estimators.
- `lotka_volterra.py`: A Bayesian Network to generate Lotka-Volterra data.
- `lotka_volterra_utils.py`: Helper functions to generate Lotka-Volterra data.
- `results.py`: Functionality to compute aggregate performance metrics and plot example posteriors.
- `settings.py`: The command line parser and a ExperimentSettings object.
- `utils.py`: A logger object and various other functions.