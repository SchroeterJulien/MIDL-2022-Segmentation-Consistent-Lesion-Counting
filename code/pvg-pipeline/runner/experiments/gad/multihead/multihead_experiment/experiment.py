# Type checks
import os
# System imports
import sys
from typing import Dict, List, Tuple, Optional

import torch

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# Required project imports
from constants.pipeline import CheckpointKeys as CP_KEYS
# required imports
import numpy as np

# user imports
from runner.experiments.gad.template.template import ExperimentMultiHeadTemplate, init_weights
from runner.models.nnunet import SplitNNUNet
from runner.experiments.gad import experiment_setup
from runner.models.downstream import Downstream  # Same model as downstream

class Hyperparameters:
    workspace: Optional[str] = None  # None links to default comet workspace
    project_name: str = experiment_setup.project_name
    progress_freq: int = 5

    num_workers: int = 4
    epochs: int = 100
    bins: List = [0, 1, 2, 3, 4]
    w1 = torch.tensor(0.001)
    w2 = torch.tensor(0.06)
    # resize if not already done
    resize: bool = False
    size: Tuple = (64, 192, 192)
    models: Dict[str, Dict] = {'unet':
                                   {'in_channels': 5,
                                    'out_channels': 1,
                                    'filters': 32,
                                    'depth': 5,
                                    'p': 0.1
                                    },
                               'multihead': {
                                   'i': 32,
                                   'num_classes': 5,
                                   'k': 64, # 64
                                   'p': [0.5, 0.25],
                                   'fc': []
                               }
                              }

    optimizers: Dict[str, Dict] = {'unet_adam' : {'lr': 1e-4, 'weight_decay': 1e-5, 'eps': 1e-8,
                                                     'betas': (0.9, 0.999)},
                                   'multihead_adam' : {'lr': 1e-5,
                                                      'weight_decay': 1e-5, 'eps': 1e-8,'betas': (0.9, 0.999)
                                                      }}
    schedulers: Dict[str, Dict] = {'unet_scheduler': {'gamma': 0.995,
                                                      'last_epoch': -1
                                                      },
                                   'multihead_scheduler': {'gamma': 0.995,
                                                      'last_epoch': -1
                                                      },
                                   }
    criterions: Dict[str, Dict] = {'unet_loss': {'weighted': True, 'pos_weight': 3},
                                   'multihead_loss': {'weighted': False, 'decaying': False,
                                                     'pos_weight': [1.0, 5.3, 11.2, 19.3, 6.9]}
    }
    cp_stat: Tuple = ('val_seg_avg_loss', 'min')
    cp_period: int = 1
    cp_patience: int = 20
    cp_save_period: int = 20

    data_augmentations = {'WrappedRandomAffine3D': {'degrees': [[-8., 8.], [-8., 8.], [-8., 8.]],
                                                    'translate': [0., 0., 0.], 'scale': [0.90, 1.10],
                                                    'shear': [[-8., 8.], [-8., 8.], [-8, 8.], [-8., 8.], [-8., 8.],
                                                              [-8., 8.]],
                                                    'p': 0.9},
                          'WrappedRandomHorizontalFlip': {'p': 0.5},
                          'WrappedRandomVerticalFlip': {'p': 0.5},
                          'WrappedRandomDepthicalFlip': {'p': 0.5}
                          }
    batch_size: int = 2
    unet_checkpoint: str = experiment_setup.unet_checkpoint


    metric_thresholds: List[float] = np.concatenate((np.linspace(-40,-10,100),np.linspace(-10,-3,100),np.linspace(-3,3,601),np.linspace(3,10,100),np.linspace(10,40,100))).tolist()


class ExperimentDescription(ExperimentMultiHeadTemplate):
    def __init__(self):
        super(ExperimentDescription, self).__init__(Hyperparameters)

    ############### IMPLEMENT FUNCTIONS BELOW #########################

    def setup_models(self)->None:
        unet = SplitNNUNet(**self.HP.models['unet'])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet
        model_checkpoint = torch.load(self.HP.unet_checkpoint, map_location='cpu')
        self.models['unet'].load_state_dict(model_checkpoint[CP_KEYS.MODELS]["unet"])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet

        # Same model as downstream
        multihead = Downstream(self.HP, **self.HP.models['multihead'])
        multihead = multihead.to(self.devices[0])
        multihead.apply(init_weights)
        self.models['multihead'] = multihead
