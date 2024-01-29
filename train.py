import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from settings import ExperimentSettings
from models import RatioEstimator
from posterior import Posterior
from utils import Logger

def train_epoch(model, optimizer, train_loader, epoch_nr, settings, logger=None):
    model.train()
    train_loss = []
    N = len(train_loader)
    for i, (x, theta, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(model.device())
        theta = theta.to(model.device())
        labels = labels.to(model.device())
        logratio = model(x, theta)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logratio, labels.float())
        loss.backward()
        if settings.training['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.training['clip_grad_norm'])
        elif loss.item() > 1e4:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss.append(loss.item())

        # Log
        if logger:
            step = epoch_nr * N + i
            logger.log_scalar('loss_it', loss.item(), step)

    return np.mean(train_loss), {'loss': np.mean(train_loss), 'losses': train_loss}

def eval_model(model, test_loader, epoch_nr, settings, logger=None):
    model.eval()
    test_loss = []
    N = len(test_loader)
    for i, (x, theta, labels) in enumerate(test_loader):
        x = x.to(model.device())
        theta = theta.to(model.device())
        labels = labels.to(model.device())
        
        logratio = model(x, theta)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logratio, labels.to(model.device()).float())
        test_loss.append(loss.item())

        # Log
        if logger:
            step = epoch_nr * N + i
            logger.log_scalar('loss_val_it', loss.item(), step)
        
    return np.mean(test_loss), {'val_loss': np.mean(test_loss), 'val_losses': test_loss}

def train_model(model: RatioEstimator, 
                train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                settings: ExperimentSettings,
                logger: Logger | None = None
               ) -> tuple[list[float], list[float], dict[str, any]]:
    """
    Trains a given model using the provided optimizer and training data, and evaluates it on the provided test data.
    
    Args:
        model (RatioEstimator): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_loader (torch.utils.data.Dataloader): The data loader for the training data.
        validation_loader (torch.utils.data.Dataloader): The data loader for the test data.
        settings (ExperimentSettings): The experiment settings.
        logger (logger | None, optional): The logger to use for logging training progress and saving models. Defaults to None.
    
    Returns:
        Tuple[List[float], List[float], Dict[str, Any]]: A tuple containing the training losses, validation losses, and the best model dictionary.
    """

    # logger.watch(model)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=settings.training['lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.training['milestones'], gamma=0.1)

    if settings.training['save_model'] != 'no' and logger == None:
        print("Warning: saving models is disabled because no logger is provided")

    evaluation_epochs = []
    if settings.training['intermediate_evaluation'] > 0:
        evaluation_epochs = np.linspace(0, settings.training['epochs'], settings.training['intermediate_evaluation']+2, dtype=int)[1:-1]
        
    
    n_epochs = settings.training['epochs']
    patience = settings.training['patience']

    # Collectors
    train_losses_batch = []
    validation_losses_batch = []
    validation_losses_epochs = []

    if settings.training['select_parameter_dims'] != 'all':
        train = (train[0], train[1][:, settings.training['select_parameter_dims']], train[2])
        validation = (validation[0], validation[1][:, settings.training['select_parameter_dims']], validation[2])

    # Construct the dataloaders for the training and validation data
    train_dataset = construct_ratio_dataset(train)
    val_dataset = construct_ratio_dataset(validation)

    train_loader = DataLoader(train_dataset, batch_size=settings.training['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.training['batch_size'], shuffle=True)

    for epoch in (pbar := tqdm(range(n_epochs), desc='Epochs', disable=not settings.logging['verbose'])):
        model.train()
        train_loss, train_info = train_epoch(model, optimizer, train_loader, epoch, settings, logger=logger)
        validation_loss, val_info = eval_model(model, val_loader, epoch, settings, logger=logger)
        train_losses_batch += train_info['losses']
        validation_losses_batch += val_info['val_losses']
        
        # Model bookkeeping
        if epoch == 0 or validation_loss < min(validation_losses_epochs):
            best_model = deepcopy(model)
            best_model_dict = {
                'model': best_model,
                'epoch':  epoch,
                'train_loss': train_loss,
                'validation_loss': validation_loss
            }
            if logger:
                logger.log(f"New best model at epoch {epoch} with validation loss {validation_loss:.4f}", verbose=False)

        validation_losses_epochs.append(validation_loss)

        # Intermediate evaluation
        if epoch in evaluation_epochs and logger:
            model.eval()
            metrics = Posterior.evaluate(best_model_dict, validation, settings)
            for key, value in metrics.items():
                # Log mean and std
                logger.log_scalar(key, value.mean(), epoch)
                logger.log_scalar(f"{key}-std", value.std(), epoch)

        # Save model
        if settings.training['save_model'] and epoch == best_model_dict['epoch'] and logger != None:
            logger.save_model(model)
        if settings.training['save_model'] == 'all' and logger != None:
            logger.save_model(model, name=f"model_{epoch}.pt")

        # Log
        if logger:
            logger.log_scalar('loss', train_loss, epoch)
            logger.log_scalar('loss_val', validation_loss, epoch)
        pbar.set_description(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Validation loss: {validation_loss:.4f}")

        # Early stopping
        if epoch > settings.training['patience'] and settings.training['patience'] > 0:
            if validation_loss > min(validation_losses_epochs[-settings.training['patience']:]):
                patience -= 1

        if patience == 0:
            if logger:
                logger.log(f"Early stopping at epoch {epoch}", verbose=False)
            pbar.close()
            break

        scheduler.step()

    return train_losses_batch, validation_losses_batch, best_model_dict

def construct_ratio_dataset(trainset):
    _, theta, x = trainset[:]
    N = x.shape[0]
    x_twice = torch.cat([x, x], dim=0)
    theta_twice = torch.cat([theta, permute_without_identity(theta)], dim=0)  # second half is shuffled
    labels = torch.cat([torch.ones((N, 1)), torch.zeros((N, 1))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(x_twice, theta_twice, labels)
    return train_dataset

def permute_without_identity(x):
    """Permute a tensor without identity mappings.

    Args:
        x (torch.Tensor): Tensor to be permuted.

    Returns:
        torch.Tensor: Permuted tensor.
    """
    # assert x.dim() >= 2

    # create a index permutation tensor
    idx = torch.randperm(x.size(0))
    # idx = torch.tensor([3, 2, 1, 0, 4])
    # create a mask of where the identity permutations are
    mask = idx == torch.arange(x.size(0))

    if mask.long().sum() == 0:
        pass
    elif mask.long().sum() == 1:
        # Swap the true element with its neighbor
        identity_idx = idx[mask]
        idx[identity_idx] = idx[identity_idx - 1]
        idx[identity_idx - 1] = identity_idx

    else:
        # Move over the identity permutation one step
        idx[mask] = torch.cat((idx[mask][1:], idx[mask][:1]), dim=0)
    x = x[idx]
    return x
