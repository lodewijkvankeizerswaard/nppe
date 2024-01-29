"""
This file contains the main function for training a model. Evaluating the
posterior is disabled by default. To enable it, uncomment the relevant lines
in the main function. 
"""

import torch.multiprocessing as mp

from settings import get_parser, process_arguments, ExperimentSettings
from utils import initialize_model_name_from_args, Logger, set_seed
from models import get_model, RatioEstimator
from lotka_volterra import get_data
from train import train_model
from posterior import Posterior

def main(settings: ExperimentSettings):
    # Set seed
    if settings.training['seed'] != 0:
        set_seed(settings.training['seed'])
    
    # Create a logger
    logger = Logger(settings)
    logger.log(str(settings), verbose=settings.logging['verbose'])    

    # Create model
    model: RatioEstimator = get_model(settings)
    model.to(settings.device)

    # Log model info
    logger.log(str(model), verbose=settings.logging['verbose'])

    # Log all settings and model hyperparameters
    logger.log_hparams(settings.flatten())

    # Get data
    train, val, test = get_data(settings, logger)

    train_losses, val_losses, best_model_dict = train_model(model, train, val, settings, logger=logger)
    logger.log(f"Best model at epoch {best_model_dict['epoch']} with validation loss {best_model_dict['validation_loss']:.4f}", verbose=settings.logging['verbose'])

    # Evaluate the posterior over alpha. We pass in data points D_i from the test set (`test_set[2]`).
    # results = Posterior.evaluate(best_model_dict, test, settings)

    # Log the metrics
    layout = {
        'best_validation_loss': best_model_dict['validation_loss'],
    }
    # for key, value in results.items():
    #     layout[f'test.{key}'] = value.mean().item()
    #     layout[f'test.{key}-std'] = value.std().item()

    logger.summary(layout=layout)
    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # Read command line arguments
    parser = get_parser()
    args = vars(parser.parse_args())

    if args['from_json']:
        print("Loading arguments from json file")
        settings = ExperimentSettings.from_json_file(args['from_json'])

        if not hasattr(settings, 'name'):
            settings.name = initialize_model_name_from_args(settings.to_args())

    else:
        # Initialize model name and process arguments
        name = initialize_model_name_from_args(args)
        args = process_arguments(args)

        bn_args, training_args, posterior_args, logging_args = args

        # Create settings object
        settings = ExperimentSettings(bn_args, training_args, posterior_args, logging_args, name)

    # Run main
    main(settings)