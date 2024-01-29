"""
This file contains functions to set the seed, to convert a matplotlib figure to
an array (for logging), and a Logger object.
"""

import torch
import numpy as np
import matplotlib.figure
import argparse
import datetime
import json
import os
import shutil

import pandas as pd
from settings import ExperimentSettings

try:
    import wandb
    wandb_message = ""
except ImportError:
    wandb_message = "Please make sure you have installed `wandb`. \
                    This is not included in the environment files."
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fig_to_array(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Convert a Matplotlib figure to a 4D numpy array with NHWC channels"""
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
    return image

def initialize_model_name_from_args(args: argparse.Namespace):
    """A function for initializing the name of the model from
       the cmd-line arguments."""
    if 'training.debug' in args:
        name = "runs/"
    else:
        name = "runs_debug/"

    if args['log.name'] != None and args['log.name'] != "":
        name += f"{args['bn.p_alpha']}/{args['bn.p_theta_alpha']}/{args['log.name']}/"
    else:
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name += f"{args['bn.p_alpha']}/{args['bn.p_theta_alpha']}/{time}/"
    return name

def dump_config(config: dict[str, any] | argparse.Namespace, path: str):
    # Convert args to dict if necessary
    if not isinstance(config, dict):
        config = vars(config)

    # Make sure the directory exists
    dir = os.path.split(path)[0]
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Dump the config
    json_string = json.dumps(config, indent=4, default=lambda o: '<not serializable>')
    with open(path, 'w') as f:
        f.write(json_string)

class Logger(object):
    """A general object to handle all logging logic, and the saving of models."""
    def __init__(self, args: argparse.Namespace):
        self.summary_called = False
        self.args = args
        self.log_type = args.logging['logger_type']

        if self.log_type == "tensorboard":
            self.writer = SummaryWriter(args.name)

        elif self.log_type == "wandb":
            if wandb_message:
                raise ImportError(wandb_message)
            os.environ["WANDB_SILENT"] = "true"

            # TODO make this modular
            wandb.init(project="thesis-ppe", entity="lodewijk", name=args.name, config=args)

        else:
            raise ValueError(f"Unknown logger type: {self.log_type}")
        # Save model directory
        self.model_directory = os.path.join(args.name, "models")
        self.data_directory = os.path.join(args.name, "data")

        # Make sure the directories exist
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # Save the config
        config_file_name = os.path.join(args.name, "config.json")
        dump_config(args, config_file_name)

        # Create a logging file location
        self.log_file = os.path.join(args.name, "log.txt")

        self.log(f"Logging to: {self.model_directory}")

    def log(self, print_string: str, verbose: bool = True):
        """A function for logging to the console and to a file."""
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print_string = f"|{time}| {print_string}"
        if verbose:
            print(print_string)
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                print_string = print_string if print_string[-1] == "\n" else print_string + "\n"
                f.write(print_string)

    def log_hparams(self, hparams: argparse.Namespace):
        if self.log_type == "tensorboard":
            self.tb_hparams = hparams
        elif self.log_type == "wandb":
            # Already done
            pass

    def log_scalar(self, name: str, value: any, timestep: int):
        if self.log_type == "tensorboard":
            self.writer.add_scalar(name, value, timestep)
        elif self.log_type == "wandb":
            wandb.log({name: value})

    def summary(self, layout: dict):
        if self.log_type == "tensorboard":
            if hasattr(self, "tb_hparams"):
                for k, v in self.tb_hparams.items():
                    if v == []:
                        self.tb_hparams[k] = "[]"
                    elif isinstance(v, list) and (isinstance(v[0], float) or isinstance(v[0], int)):
                        self.tb_hparams[k] = torch.tensor(v)
                    elif isinstance(v, list) and isinstance(v[0], str):
                        self.tb_hparams[k] = str(v)
                self.writer.add_hparams(self.tb_hparams, layout)
            else:
                self.log("WARNING: No hyperparameters logged, please call `log_hparams` first")
        elif self.log_type == "wandb":
            for name, value in layout.items():
                wandb.run.summary[name] = value

        self.summary_called = True

    def log_image(self, name: str, image: any, timestep: int = 0):
        if self.log_type == "tensorboard":
            self.writer.add_image(name, image, dataformats="NHWC", global_step=timestep)
        elif self.log_type == "wandb":
            img = wandb.Image(image, caption=name)
            wandb.log({name: img})

    def log_images(self, image_dict: dict[str, any], timestep: int | list[int] | None = None):
        if isinstance(timestep, list):
            assert len(timestep) == len(image_dict), "Timestep and image_dict must have the same size"
        else:
            timestep = [timestep] * len(image_dict)

        if self.log_type == "tensorboard":
            for (name, image), t in zip(image_dict.items(), timestep):
                self.writer.add_image(name, image, dataformats="NHWC", global_step=t)
        elif self.log_type == "wandb":
            for (name, image), t in zip(image_dict.items(), timestep):
                img = wandb.Image(image, caption=name)
                wandb.log({name: img})

    def save_data(self, name: str, data: torch.Tensor):
        data_file_name = os.path.join(self.data_directory, f"{name}.pt")
        torch.save(data, data_file_name)

    def load_data(self, name: str) -> torch.Tensor:
        data_file_name = os.path.join(self.data_directory, f"{name}.pt")
        return torch.load(data_file_name)

    def save_dataframe(self, name: str, df: pd.DataFrame):
        data_file_name = os.path.join(self.data_directory, f"{name}.csv")
        df.to_csv(data_file_name)

    def save_model(self, model: torch.nn.Module, name: str = "model.pt"):
        model_file_name = os.path.join(self.model_directory, name)
        torch.save(model.state_dict(), model_file_name)

    def watch(self, model: torch.nn.Module):
        if self.log_type == "wandb":
            wandb.watch(model)
        elif self.log_type == "tensorboard":
            self.writer.add_graph(model)

    def __del__(self):
        if not self.summary_called:
            self.summary({})
        if self.log_type == "tensorboard":
            self.writer.close()

    @staticmethod
    def load_model(args: argparse.Namespace) -> torch.nn.Module:
        model_file_name = os.path.join(args.name, "models", "model.pt")
        model = torch.load(model_file_name)
        return model