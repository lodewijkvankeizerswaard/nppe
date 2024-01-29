"""
This file contains code to manage command line arguments, ExperimentSettings and 
parameter contraints for the Lotka-Volterra simulator.
"""

import argparse
import torch
import json
import sys
import pprint

hard_constraints = {
    'alpha_upper_bound': 1.7,
    'alpha_lower_bound': 1.1,
    'beta_upper_bound': 0.9,
    'beta_lower_bound': 0.1,
    'gamma_upper_bound': 0.7,
    'gamma_lower_bound': 0.1,
    'delta_upper_bound': 0.7,
    'delta_lower_bound': 0.1,
    'x0_upper_bound': 100,
    'x0_lower_bound': 1,
    'y0_upper_bound': 100,
    'y0_lower_bound': 1
}

def get_constraint_map(interval: tuple[float, float], select_axis: list, inverse: bool = False):
    """
    Generate a mapping function that maps values from the given interval to the corresponding constraints.

    Args:
        interval (tuple): A tuple representing the interval [a, b] to be mapped.
        select_axis (list): A list of indices specifying the axes to be selected for mapping.
        inverse (bool, optional): If True, the mapping is performed from the constrained 
                                  space, to the given interval. Defaults to False.

    Returns:
        function: A mapping function that takes a tensor as input and returns the mapped tensor.

    """

    if not inverse:
        a = torch.tensor([interval[0]]).repeat(len(select_axis))
        b = torch.tensor([interval[1]]).repeat(len(select_axis))

        ordered_axis_names = ['alpha', 'beta', 'gamma', 'delta', 'x0', 'y0']
        c = torch.tensor([hard_constraints[f'{axis}_lower_bound'] for axis in ordered_axis_names])[select_axis]
        d = torch.tensor([hard_constraints[f'{axis}_upper_bound'] for axis in ordered_axis_names])[select_axis]
    else:
        ordered_axis_names = ['alpha', 'beta', 'gamma', 'delta', 'x0', 'y0']
        a = torch.tensor([hard_constraints[f'{axis}_lower_bound'] for axis in ordered_axis_names])[select_axis]
        b = torch.tensor([hard_constraints[f'{axis}_upper_bound'] for axis in ordered_axis_names])[select_axis]

        c = torch.tensor([interval[0]]).repeat(len(select_axis))
        d = torch.tensor([interval[1]]).repeat(len(select_axis))

    # Map from [a, b] to [c, d] for each parameter
    add = c
    scale = (torch.eye(len(select_axis)) * (d - c) / (b - a))
    shift = a

    mapping = lambda x: add.to(x.device) + (x - shift.to(x.device)) @ scale.to(x.device)

    return mapping


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_json', type=str, default=None, help="Path to a json file containing the arguments. If set, all other arguments are ignored.")

    ## Bayesian network parameters
    parser.add_argument('--bn.p_alpha', type=str, choices=['uniform', 'continuousbern', 'gaussian'], default='uniform')
    parser.add_argument('--bn.p_alpha_args', type=str, default='{}')
    parser.add_argument('--bn.p_theta_alpha', type=str, choices=['continuousbern', 'gaussian', 'gaussian-means'], default='gaussian-means')
    parser.add_argument('--bn.p_theta_alpha_args', type=str, default='{}')
    parser.add_argument('--bn.M', type=int, default=100)
    parser.add_argument('--bn.dt', type=float, default=0.1)
    parser.add_argument('--bn.x_noise_level', type=float, default=1)
    parser.add_argument('--bn.save_data', type=bool, default=True)
    parser.add_argument('--bn.load_debug_data', action='store_true')
    parser.add_argument('--bn.train_N', type=int, default=100)
    parser.add_argument('--bn.val_N', type=int, default=100)
    parser.add_argument('--bn.test_N', type=int, default=100)
    parser.add_argument('--bn.train_I', type=int, default=1)
    parser.add_argument('--bn.train_J', type=int, default=1)
    parser.add_argument('--bn.val_I', type=int, default=1)
    parser.add_argument('--bn.val_J', type=int, default=1)
    parser.add_argument('--bn.test_I', type=int, default=1)
    parser.add_argument('--bn.test_J', type=int, default=1)

    ## Training parameters
    parser.add_argument('--training.model', type=str, choices=['cnn'], default='cnn')
    parser.add_argument('--training.select_parameter_dims', type=str, default='all', help="Set to 'all' to select all parameters, or pass a list of interger indices to select specific parameters.")
    parser.add_argument('--training.seed', type=int, default=0)
    parser.add_argument('--training.hidden_size', type=int, default=32)
    parser.add_argument('--training.hidden_layers_data', type=int, default=4)
    parser.add_argument('--training.data_channels', type=str, nargs='+', default=['diff'], choices=['diff', 'log', 'rec', 'exp', 'diff_inter'])
    parser.add_argument('--training.hidden_layers_param', type=int, default=4)
    parser.add_argument('--training.hidden_layers_out', type=int, default=4)
    parser.add_argument('--training.lr', type=float, default=0.001)
    parser.add_argument('--training.epochs', type=int, default=100)
    parser.add_argument('--training.batch_size', type=int, default=1024)
    parser.add_argument('--training.device', type=str, default='cuda')
    parser.add_argument('--training.nr_threads', type=int, default=10)
    parser.add_argument('--training.debug', action='store_true')
    parser.add_argument('--training.save_model', type=str, choices=['no', 'best', 'all'], default='best')
    parser.add_argument('--training.patience', type=int, default=-1, help="Set to -1 to disable early stopping")
    parser.add_argument('--training.clip_grad_norm', type=float, default=0, help="Set to 0 to disable gradient clipping")
    parser.add_argument('--training.milestones', type=int, nargs='+', default=[])
    parser.add_argument('--training.param_dropout', type=float, default=0, help='Set to 0 to disable parameter dropout')
    parser.add_argument('--training.intermediate_evaluation', type=int, default=0, help="Amount of intermediate evaluations during training. Set to 0 to disable.")
    # parser.add_argument('--training.retrain', action='store_true', help="Set to True to retrain a model in `log.name`")

    ## Posterior arguments
    parser.add_argument('--posterior.p_alpha', type=str, choices=['uniform', 'grid'], default='uniform')
    parser.add_argument('--posterior.p_theta_alpha', type=str, choices=['continuousbern', 'gaussian-means'], default='continuousbern')
    parser.add_argument('--posterior.p_theta_alpha_sigma', type=float, default=0.1)
    parser.add_argument('--posterior.resolution', type=int, default=5)
    parser.add_argument('--posterior.K', type=int, default=10)
    parser.add_argument('--posterior.test_points', type=int, default=10)
    parser.add_argument('--posterior.save_data', action='store_true')
    parser.add_argument('--posterior.log_posterior_plots', action='store_true')

    ## Logging parameters
    parser.add_argument('--log.logger-type', type=str, choices=['tensorboard', 'wandb'], default='tensorboard')
    parser.add_argument('--log.verbose', action='store_true')
    parser.add_argument('--log.name', type=str)
    parser.add_argument('--log.runid', type=str, default=None)
    return parser

def process_arguments(args):
    """
    Processes the arguments passed to the function and returns a dictionary of arguments
    for each of the following categories: Bayesian Network (generation of the data), training, plotting, logging, and modeling.

    Args:
        args (dict): A dictionary of arguments.

    Returns:
        tuple: A tuple of dictionaries containing the arguments for each category.
    """
    bn_args = {}
    training_args = {}
    posterior_args = {}
    logging_args = {}

    for key, value in args.items():
        if key.endswith('_args'):
            value = json.loads(value)

        if key.startswith('bn.'):
            bn_args[key[3:]] = value
        elif key.startswith('training.'):
            training_args[key[9:]] = value
        elif key.startswith('posterior.'):
            posterior_args[key[10:]] = value
        elif key.startswith('log.'):
            logging_args[key[4:]] = value
        else:
            print(f'Unknown argument: {key}')

    if training_args['select_parameter_dims'] != 'all':
        training_args['select_parameter_dims'] = [int(c) for l in training_args['select_parameter_dims'].split(',') for c in l  if c.isdigit()]

    return bn_args, training_args, posterior_args, logging_args

def dump_default_parser(args):
    parser = get_parser()
    bn, training, posterior, logging = process_arguments(vars(parser.parse_args(args)))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bn)
    pp.pprint(training)
    pp.pprint(posterior)
    pp.pprint(logging)

class ExperimentSettings:
    def __init__(self, bn: dict, training: dict, posterior: dict, logging: dict, name: str):
        global hard_constraints

        self.bn = bn
        self.constraints = hard_constraints
        self.training = training
        self.posterior = posterior
        self.logging = logging
        self.name = name

        if 'cuda' in self.training['device'] and torch.cuda.is_available():
            # print(f"Using CUDA: {self.training['device']}")
            self.device = torch.device(self.training['device'])
        else:
            # print("Using CPU")
            self.device = torch.device('cpu')

    def _hash_settings(self) -> str:
        return str(hash(json.dumps(self.bn, sort_keys=True, ensure_ascii=True)))

    def flatten(self) -> dict:
        d = {}
        flatten_dict(self.bn, d, f'bn')
        flatten_dict(self.constraints, d, f'constraints')
        flatten_dict(self.training, d, f'training')
        flatten_dict(self.posterior, d, f'posterior')
        return d
    
    def __str__(self) -> str:
        s =  f"--------------------\n" + \
             f"|ExperimentSettings|\n" + \
             f"--------------------\n" + \
             f"bn: {json.dumps(self.bn, indent=4, sort_keys=True)}\n" + \
             f"constraints: {json.dumps(self.constraints, indent=4, sort_keys=True)}\n" + \
             f"training: {json.dumps(self.training, indent=4, sort_keys=True)}\n" + \
             f"posterior: {json.dumps(self.posterior, indent=4, sort_keys=True)}\n" + \
             f"logging: {json.dumps(self.logging, indent=4, sort_keys=True)}\n" + \
             f"device: {self.device}\n"

        return s
    
    @staticmethod
    def map_thetas(theta_samples: torch.tensor, constr: dict, select_axis=None) -> torch.tensor:
            """
            Maps the theta samples from [0,1] to the constrained parameter space. If a smaller range is desired,
            make sure to affine transform the samples to a subspace of [0,1] before calling this function.

            Args:
                    theta_samples (torch.tensor): The theta samples to be mapped.
                    constr (dict): A dictionary containing the upper and lower bounds for each parameter.

            Returns:
                    torch.tensor: The mapped theta samples.
            """
            if select_axis != None and select_axis != 'all':
                # define current upper and lower bounds
                param_count = len(select_axis)
                clb = torch.zeros(param_count)
                cub = torch.ones(param_count)

                ordered_axis_names = ['alpha', 'beta', 'gamma', 'delta', 'x0', 'y0']
                dub = torch.tensor([constr[f'{param}_upper_bound'] for param in ordered_axis_names])[select_axis]
                dlb = torch.tensor([constr[f'{param}_lower_bound'] for param in ordered_axis_names])[select_axis]

                # compute affine transformation
                add = dlb.to(theta_samples.device)
                scale = (torch.eye(param_count) * (dub - dlb) / (cub - clb)).to(theta_samples.device)
                shift = clb.to(theta_samples.device)

                # apply affine transformation
                theta_samples = add + (theta_samples - shift) @ scale
            else:
                theta_samples[:,:,0] = theta_samples[:,:,0] * (constr['alpha_upper_bound'] - constr['alpha_lower_bound']) + constr['alpha_lower_bound']
                theta_samples[:,:,1] = theta_samples[:,:,1] * (constr['beta_upper_bound'] - constr['beta_lower_bound']) + constr['beta_lower_bound']
                theta_samples[:,:,2] = theta_samples[:,:,2] * (constr['gamma_upper_bound'] - constr['gamma_lower_bound']) + constr['gamma_lower_bound']
                theta_samples[:,:,3] = theta_samples[:,:,3] * (constr['delta_upper_bound'] - constr['delta_lower_bound']) + constr['delta_lower_bound']
                theta_samples[:,:,4] = theta_samples[:,:,4] * (constr['x0_upper_bound'] - constr['x0_lower_bound']) + constr['x0_lower_bound']
                theta_samples[:,:,5] = theta_samples[:,:,5] * (constr['y0_upper_bound'] - constr['y0_lower_bound']) + constr['y0_lower_bound']

            return theta_samples
    
    def to_args(self) -> argparse.Namespace:
        args = argparse.Namespace()
        for key, value in self.bn.items():
            setattr(args, f'bn.{key}', value)
        for key, value in self.training.items():
            setattr(args, f'training.{key}', value)
        for key, value in self.posterior.items():
            setattr(args, f'posterior.{key}', value)
        for key, value in self.logging.items():
            setattr(args, f'log.{key}', value)
        return args
    
    @staticmethod
    def from_json_file(path: str):
        with open(path) as f:
            config = json.load(f)

        bn_args = {}
        training_args = {}
        posterior_args = {}
        logging_args = {}

        bn_args = config['bn']
        training_args = config['training']
        constraints = config['constraints']
        posterior_args = config['posterior']
        logging_args = config['logging']
        name = config['name']

        settings = ExperimentSettings(bn_args, training_args, posterior_args, logging_args, name)
        settings.constraints = constraints
        return settings
        
    
def flatten_dict(input_dict: dict, output_dict: dict = {}, prefix: str = '') -> None:
    for key, value in input_dict.items():
        new_key = prefix + '.' + key if prefix else key
        if not isinstance(value, dict):
            output_dict[new_key] = value
        else:
            flatten_dict(value, output_dict, new_key)

if __name__ == "__main__":
    dump_default_parser(sys.argv[1:])