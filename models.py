"""
This file contains the model architecture for the ratio estimator.
"""

import torch
import numpy as np

from settings import ExperimentSettings, get_constraint_map

class RatioEstimator(torch.nn.Module):
    total_params = None
    trainable_params = None
    def __init__(self, select_parameter_dims: list[int] | None = None):
        super().__init__()

        # self._normalize_theta = get_constraint_map([-.5, .5], select_axis=select_parameter_dims)
        # self._normalize_x = get_constraint_map([-1, 1], select_axis=[4,5])

    def device(self):
        return next(self.parameters()).device
    
    def predict_labels(self, X, theta):
        X = X.to(self.device())
        theta = theta.to(self.device())
        logratio = self.forward(X, theta)
        logits = torch.sigmoid(logratio)
        preds = logits > 0.5
        return preds.long()

    # def _normalize_theta(self, theta):
        # return self.add + (theta + self.shift) @ self.scale
    
    def process(self, X: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # theta = self._normalize_theta(theta)
        # X = self._normalize_x(X.transpose(1, 2)).transpose(1, 2)
        return X, theta
    
    def forward(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def get_model(settings: ExperimentSettings) -> RatioEstimator:
    """
    Returns a ratio estimator model based on the given experiment settings.

    Args:
        settings (ExperimentSettings): The experiment settings.

    Returns:
        RatioEstimator: The ratio estimator model.
    """

    theta_size = len(settings.training['select_parameter_dims']) if settings.training['select_parameter_dims'] != 'all' else 6
    
    if settings.training['model'] == 'mlp':
        raise NotImplementedError
    elif settings.training['model'] == 'cnn':
        model = CNNRE(data_shape=(2, settings.bn['M']), 
                      data_channels=settings.training['data_channels'],
                      param_shape=(theta_size, ), 
                      hidden_size=settings.training['hidden_size'], 
                      hidden_layers_data=settings.training['hidden_layers_data'],
                      hidden_layers_param=settings.training['hidden_layers_param'],
                      hidden_layers_out=settings.training['hidden_layers_out'],
                      select_parameter_dims=settings.training['select_parameter_dims'],
                      param_dropout=settings.training['param_dropout'])
    else:            
        raise ValueError(f"Unknown model {settings.training['model']}")
    
    return model

    
class ConvDebug(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X

class CNNRE(RatioEstimator):
    def __init__(self, data_shape: tuple, 
                 data_channels: list[str],
                 param_shape: tuple, 
                 hidden_size: int = 32, 
                 hidden_layers_data: int = 4, 
                 hidden_layers_param: int = 4, 
                 hidden_layers_out: int = 4,  
                 select_parameter_dims: list[int] | None = None,
                 param_dropout: float = 0.2):
        super().__init__(select_parameter_dims)
        self.data_shape = data_shape
        self.param_shape = param_shape

        self.hidden_size = hidden_size

        self.param_data_attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)


        self.theta_dropout = torch.nn.Dropout(p=param_dropout)
        self.theta_mlp = torch.nn.Sequential(*[
            torch.nn.Linear(np.prod(param_shape), hidden_size), torch.nn.ReLU(),
            *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for _ in range(hidden_layers_param - 1)],
        ])

        
        self._channel_transforms = self._channel_list(data_channels)
        self.convs = torch.nn.Sequential(*[ConvDebug(),
                                           torch.nn.Conv2d(in_channels=1 + len(self._channel_transforms), 
                                                          out_channels=self.hidden_size//4,
                                                          kernel_size=(2, 5), 
                                                          stride=(2, 3), 
                                                          padding=(1, 2),
                                                          padding_mode='circular'),
                                          torch.nn.ReLU(),
                                          ConvDebug(),
                                          # inplace convolution (i.e. does not change input shape)
                                          torch.nn.Conv2d(in_channels=hidden_size//4, 
                                                          out_channels=hidden_size//4, 
                                                          kernel_size=(2, 3), 
                                                          stride=(2, 1), 
                                                          padding=(1, 1),
                                                          padding_mode='circular'),
                                          torch.nn.ReLU(), 
                                          ConvDebug(),
                                          # inplace convolution (i.e. does not change input shape)
                                          torch.nn.Conv2d(in_channels=hidden_size//4, 
                                                          out_channels=hidden_size//4,
                                                          kernel_size=(2, 3),
                                                            stride=(2, 1),
                                                            padding=(1, 1),
                                                            padding_mode='circular'),
                                          torch.nn.ReLU(),
                                          # inplace convolution (i.e. does not change input shape)
                                          torch.nn.Conv2d(in_channels=hidden_size//4, 
                                                          out_channels=hidden_size//4,
                                                          kernel_size=(2, 3),
                                                            stride=(2, 1),
                                                            padding=(1, 1),
                                                            padding_mode='circular'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(in_channels=hidden_size//4, 
                                                          out_channels=hidden_size//2, 
                                                          kernel_size=(2, 5), 
                                                          stride=(2, 3), 
                                                          padding=(1, 2),
                                                          padding_mode='circular'),
                                          torch.nn.ReLU(),
                                          ConvDebug(),
                                          # inplace convolution (i.e. does not change input shape)
                                          torch.nn.Conv2d(in_channels=hidden_size//2, 
                                                          out_channels=hidden_size//2, 
                                                          kernel_size=(2, 3), 
                                                          stride=(2, 1), 
                                                          padding=(1, 1),
                                                          padding_mode='circular'),
                                          torch.nn.ReLU(), 
                                          ConvDebug(),
                                          # inplace convolution (i.e. does not change input shape)
                                          torch.nn.Conv2d(in_channels=hidden_size//2, 
                                                          out_channels=hidden_size//2, 
                                                          kernel_size=(2, 3), 
                                                          stride=(2, 1), 
                                                          padding=(1, 1),
                                                          padding_mode='circular'),
                                          torch.nn.ReLU(), 
                                          # inplace convolution (i.e. does not change input shape)
                                          torch.nn.Conv2d(in_channels=hidden_size//2, 
                                                          out_channels=hidden_size//2, 
                                                          kernel_size=(2, 3), 
                                                          stride=(2, 1), 
                                                          padding=(1, 1),
                                                          padding_mode='circular'),
                                          torch.nn.ReLU(), 
                                          ConvDebug(),
                                          torch.nn.Conv2d(in_channels=hidden_size//2, 
                                                          out_channels=hidden_size, 
                                                          kernel_size=(2, 5), 
                                                          stride=(1,3), 
                                                          padding=(0,1),
                                                          padding_mode='circular'),
                                          torch.nn.ReLU(),
                                          ConvDebug(),
                                          
                                          torch.nn.Conv2d(in_channels=hidden_size, 
                                                          out_channels=hidden_size, 
                                                          kernel_size=(1, 4), 
                                                          stride=1, 
                                                          padding=0),
                                          torch.nn.ReLU(),
        ])
        self.conv_mlp = torch.nn.Sequential(*[
            *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for _ in range(hidden_layers_data)],
        ])

        self.out_mlp = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_size * 2, hidden_size), torch.nn.ReLU(),
            *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for _ in range(hidden_layers_out)],
            torch.nn.Linear(hidden_size, 1),
        ])

    def _channel_list(self, channel_args):
        channel_transforms = []
        if 'diff' in channel_args:
            channel_transforms.append(lambda X: torch.nn.functional.pad(X[:, :, 1:] - X[:, :, :-1], (0, 1)).unsqueeze(dim=1))

        if 'log' in channel_args:
            channel_transforms.append(lambda X: torch.log(X.abs() + 1e-6).unsqueeze(dim=1))

        if 'rec' in channel_args:
            channel_transforms.append(lambda X: (1 / X).unsqueeze(dim=1))

        if 'exp' in channel_args:
            channel_transforms.append(lambda X: torch.exp(X).unsqueeze(dim=1))

        if 'diff_inter' in channel_args:
            channel_transforms.append(lambda X: torch.nn.functional.pad(X[:, [1, 0], 1:] - X[:, :, :-1], (0, 1)).unsqueeze(dim=1))

        return channel_transforms

    def add_data_channels(self, X):
        X_transforms = [transform(X) for transform in self._channel_transforms]
        X = X.unsqueeze(dim=1)
        X = torch.cat([X] + X_transforms, dim=1)
        return X


    def forward(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        X, theta = super().process(X, theta)
        # Input shapes
        # X: (B, 2, M)
        # theta: (B, 6)

        # Output shape
        # logratio: (B, 1)

        X = self.add_data_channels(X)
        X = self.convs(X).squeeze(dim=-1).squeeze(dim=-1)
        theta = self.theta_dropout(theta)
        theta = self.theta_mlp(theta)
        outputs = torch.cat([X, theta], dim=1)
        logratio = self.out_mlp(outputs)
        return logratio
