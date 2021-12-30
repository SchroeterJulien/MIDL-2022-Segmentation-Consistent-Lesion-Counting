# Type checks
from typing import Dict, List, Tuple, Optional

# System imports
import sys
import os

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# required imports
import numpy as np

from runner.experiments.gad.template.template import ExperimentTemplate
from runner.experiments.gad import experiment_setup

class Hyperparameters:
    workspace: Optional[str] = None  # None links to default comet workspace
    project_name: str = experiment_setup.project_name
    progress_freq: int = 5

    batch_size: int = 4

    num_workers: int = 4
    epochs: int = 100
    # resize if not already done
    resize: bool = False
    size: Tuple = (64, 192, 192)
    models: Dict[str, Dict] = {'unet': {'in_channels': 5,
                                        'out_channels': 1,
                                        'filters': 32,
                                        'depth': 5,
                                        'p': 0.1
                                        }
                               }
    optimizers: Dict[str, Dict] = {'unet_adam': {'lr': 1e-4, 'weight_decay': 1e-5, 'eps': 1e-8, 'betas': (0.9, 0.999)}}
    schedulers: Dict[str, Dict] = {}
    criterions: Dict[str, Dict] = {'unet_loss': {'weighted': True, 'decaying': False, 'pos_weight': 3,
                                                 'decay_factor': 0.97}}

    data_augmentations = {'WrappedRandomAffine3D': {'degrees': [[-8., 8.], [-8., 8.], [-8., 8.]],
                                                    'translate': [0., 0., 0.], 'scale': [0.90, 1.10],
                                                    'shear': [[-8., 8.], [-8., 8.], [-8, 8.], [-8., 8.], [-8., 8.],
                                                              [-8., 8.]],
                                                    'p': 0.9},
                          'WrappedRandomHorizontalFlip': {'p': 0.5},
                          'WrappedRandomVerticalFlip': {'p': 0.5},
                          'WrappedRandomDepthicalFlip': {'p': 0.5}
                          }

    cp_stat: Tuple = ('val_avg_loss', 'min')
    cp_period: int = 1
    cp_patience: int = 1
    cp_save_period: int = 20

    metric_thresholds: List[float] = np.concatenate((np.linspace(-40, -10, 100), np.linspace(-10, -3, 100),
                                                     np.linspace(-3, 3, 601), np.linspace(3, 10, 100),
                                                     np.linspace(10, 40, 100))).tolist()


class ExperimentDescription(ExperimentTemplate):
    def __init__(self):
        super(ExperimentDescription, self).__init__(Hyperparameters)





