# Type checks
import os
# System imports
import sys
from typing import Dict, List, Tuple, Union, Optional

import torch
from ignite.engine import Engine
from torch import Tensor

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# Required project imports
from constants.pipeline import CheckpointKeys as CP_KEYS
from ignite.metrics import Average, Accuracy
from runner.builder.assembly import MetricAssembler
from runner.metrics.metric import BinaryThresholdConfusionMatrix, ApproximateMetrics, btcmF1

# required imports
import numpy as np
from scipy import ndimage

# user imports
from runner.experiments.gad.template.template import ExperimentTemplate
from runner.transforms.augmentation import CustomTransforms
from runner.models.nnunet import NNUNet
from runner.experiments.gad import experiment_setup


class Hyperparameters:
    workspace: Optional[str] = None # None links to default comet workspace
    project_name: str = experiment_setup.project_name
    progress_freq: int = 5
    cc_threshold: float = 0.1
    num_classes: int = 5
    num_extra_masks = 0
    pad = 2
    extra_size = 1 + pad
    w1 = torch.tensor(0.001)
    w2 = torch.tensor(0.06)

    batch_size: int = 2
    unet_checkpoint: str = experiment_setup.unet_checkpoint

    num_workers: int = 4
    epochs: int = 200

    # resize if not already done
    resize: bool = False
    size: Tuple = (64, 192, 192)

    bins: List = [0, 1, 2, 3, 4]
    models: Dict[str, Dict] = {'unet':
                                   {'in_channels': 5,
                                    'out_channels': 1,
                                    'filters': 32,
                                    'depth': 5,
                                    'p': 0.1
                                    }
                              }
    optimizers: Dict[str, Dict] = {'unet_adam' : {'lr': 1e-4, 'weight_decay': 1e-5, 'eps': 1e-8,
                                                     'betas': (0.9, 0.999)}}
    schedulers: Dict[str, Dict] = {'unet_scheduler': {'gamma': 0.995,
                                                      'last_epoch': -1
                                                      }}
    criterions: Dict[str, Dict] = {'unet_loss': {'weighted': True, 'decaying': False, 'pos_weight': 3.0,
                                                 'decay_factor': 0.97}}
    cp_stat: Tuple = ('val_seg_avg_loss', 'min')
    cp_period: int = 1
    cp_patience: int = 10
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

    metric_thresholds: List[float] = np.concatenate((np.linspace(-40,-10,100),np.linspace(-10,-3,100),np.linspace(-3,3,601),np.linspace(3,10,100),np.linspace(10,40,100))).tolist()


class ExperimentDescription(ExperimentTemplate):
    def __init__(self):
        super(ExperimentDescription, self).__init__(Hyperparameters)
        self.cca_structure = ndimage.generate_binary_structure(3, 2)

        if self.HP.bins is not None:
            self.bin = CustomTransforms.BinCount(bins=self.HP.bins)
        else:
            self.bin = None
        self.i = 0

    def setup_criterions(self) -> None:
        super().setup_criterions()

    ############### IMPLEMENT FUNCTIONS BELOW #########################

    def train_batching(self, batch: Union[Dict, Tensor])->Union[Dict, List, Tensor, Tuple]:
        # Cast labels to float
        batch['LABEL'] = batch['LABEL'].float()

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        if self.bin is not None:
            batch['COUNT'] = self.bin(batch['COUNT']).to(self.devices[0])

        # Pad the input to be an even size
        if self.HP.resize:
            batch = CustomTransforms.resize(batch, size=(64, 192, 192),
                                            imaging_keys=['MRI', 'LABEL', 'MASK'])  # CustomTransforms.pad_even(batch)

        # Augment
        for augmentation in self.augmentations:
            batch = augmentation(batch)

        return batch

    def evaluate_batching(self, batch: Union[Dict, Tensor])->Union[Dict, List, Tensor, Tuple]:
        # Cast labels to float
        batch['LABEL'] = batch['LABEL'].float()

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        if self.bin is not None:
            batch['COUNT'] = self.bin(batch['COUNT']).to(self.devices[0])

        if self.HP.resize:
            batch = CustomTransforms.resize(batch, size=(64, 192, 192), imaging_keys=['MRI', 'LABEL', 'MASK'])

        return batch

    def setup_models(self)->None:
        unet = NNUNet(**self.HP.models['unet'])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet
        model_checkpoint = torch.load(self.HP.unet_checkpoint, map_location='cpu')
        self.models['unet'].load_state_dict(model_checkpoint[CP_KEYS.MODELS]["unet"])


    def forward(self, batch, mode='train', epoch=None):
        unet = self.models["unet"]
        if mode == 'train':
            unet.train()
        else:
            unet.eval()
        seg = unet(batch["MRI"])

        # Apply sigmoid to the segmentation predictions for counting (automatically included in segmentation loss
        # so use a clone)
        seg_sigmoid = torch.sigmoid(seg.clone())
        count_predictions, cc_count = self.get_label_masks(seg_sigmoid, batch, epoch)
        return seg, count_predictions, cc_count

    def calculate_count_loss(self, count_pred, batch):
        loss_count = -1*torch.mean(torch.log(count_pred + 1e-12) * batch['COUNT'])
        return loss_count

    def calculate_losses(self, preds, batch, epoch=None):
        seg_loss = super().calculate_losses(preds[0], batch)
        count_loss = self.calculate_count_loss(preds[1], batch)
        return seg_loss, count_loss

    def backward(self, loss, epoch=None):
        unet_optimizer = self.optimizers['unet_adam']
        unet_optimizer.zero_grad()

        weight = self.HP.w1 * torch.exp(self.HP.w2 * epoch)
        total_loss = loss[0] + weight * loss[1]

        total_loss.backward()
        unet_optimizer.step()

    def prepare_output(self, preds, loss, batch):
        seg = preds[0]
        pb_count = preds[1]
        cc_count = preds[2]
        seg_loss = loss[0]
        count_loss = loss[1]
        one_hot = torch.zeros(size=(pb_count.size(0), pb_count.size(1)), device=self.devices[0])
        one_hot[torch.arange(pb_count.size(0)), torch.argmax(pb_count, dim=1)] = 1.0

        return {'preds': seg[batch['MASK'].bool()].detach(),
                'label': batch['LABEL'][batch['MASK'].bool()].detach(),
                'count': torch.argmax(batch['COUNT'], dim=1).detach(),
                'pb_count': one_hot.detach(),
                'cc_count': torch.argmax(self.bin(cc_count), dim=1).to(self.devices[0]).detach(),
                'cc_count_bin': self.bin(cc_count).to(self.devices[0]).detach(),
                'seg_loss': seg_loss.item(),
                'count_loss': count_loss.item()
                }

    def setup_train_metrics(self) ->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['seg_loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'train_seg_avg_loss')

        avg_count_loss_metric = Average(output_transform=lambda output: output['count_loss'], device=self.devices[0])
        assembler.add(avg_count_loss_metric, 'train_count_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds',
                                              label_key='label', device=self.devices[0])
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'train_PR_AUC')
        assembler.add(ROC_AUC, 'train_ROC_AUC')
        assembler.add(F1, 'train_F1')

        # Accuracy
        acc = Accuracy(output_transform=lambda output: (output['pb_count'], output['count']), device=self.devices[0])
        assembler.add(acc, 'train_acc')
        acc = Accuracy(output_transform=lambda output: (output['pb_count'], output['cc_count']), device=self.devices[0])
        assembler.add(acc, 'train_consistency_acc')
        acc = Accuracy(output_transform=lambda output: (output['cc_count_bin'], output['count']), device=self.devices[0])
        assembler.add(acc, 'train_cc_acc')

        self.train_metrics = assembler.build()

    def setup_evaluate_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['seg_loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'val_seg_avg_loss')

        avg_count_loss_metric = Average(output_transform=lambda output: output['count_loss'], device=self.devices[0])
        assembler.add(avg_count_loss_metric, 'val_count_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds',
                                              label_key='label', device=self.devices[0])
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'val_PR_AUC')
        assembler.add(ROC_AUC, 'val_ROC_AUC')
        assembler.add(F1, 'val_F1')

        # Accuracy
        acc = Accuracy(output_transform=lambda output: (output['pb_count'], output['count']), device=self.devices[0])
        assembler.add(acc, 'val_acc')
        acc = Accuracy(output_transform=lambda output: (output['pb_count'], output['cc_count']), device=self.devices[0])
        assembler.add(acc, 'val_consistency_acc')
        acc = Accuracy(output_transform=lambda output: (output['cc_count_bin'], output['count']), device=self.devices[0])
        assembler.add(acc, 'val_cc_acc')

        self.evaluate_metrics = assembler.build()

    ##################################################################

    def get_label_masks(self, preds, batch, epoch, max_instance=100):
        # apply the brain mask
        # now take a copy so as not to track the binarization
        # use torch.no_grad to be sure no gradients are added to the computational graph
        with torch.no_grad():
            preds_copy = preds.detach().clone()
            preds_copy[~batch['MASK'].bool()] = 0.0
            preds_copy.requires_grad = False
            preds_copy = torch.where(preds_copy > self.HP.cc_threshold, 1.0, 0.0)

        # all_masks (batch, max_instance, D, H, W)
        all_masks = torch.zeros(size=(preds.size(0), max_instance, preds.size(2), preds.size(3), preds.size(4)),
                                device=self.devices[0], requires_grad=False)
        cc_counts = torch.zeros(size=(preds.size(0), 1), device=self.devices[0])
        for i, pred in enumerate(preds_copy):
            # use torch.no_grad to be sure no gradients are added to the computational graph
            with torch.no_grad():
                # use connected components to get regions
                output_label, output_num_features = ndimage.label(pred.squeeze(0).detach().cpu().numpy(),
                                                                  structure=self.cca_structure)
                cc_counts[i] = output_num_features

                output_label = torch.from_numpy(output_label)
                output_label = output_label.to(self.devices[0])

            for label in torch.unique(output_label):
                # don't take maximum from background class
                if label > 0:
                    # use torch.no_grad to be sure no gradients are added to the computational graph
                    with torch.no_grad():
                        mask = torch.where(output_label == label, 1.0, 0.0)

                    # gradient should be tracked here
                    all_masks[i, label-1, :, :, :] = mask

        all_max = torch.amax(all_masks * preds, dim=(2, 3, 4))

        count_predictions = self.Counting(all_max, max_occurence=self.HP.num_classes)
        count_predictions = count_predictions.to(self.devices[0])

        return count_predictions, cc_counts

    def Counting(self, pred, max_occurence=5):
        contribution = torch.unbind(pred, 1)

        count_prediction = torch.cuda.FloatTensor(pred.size()[0], max_occurence).fill_(0)
        count_prediction[:, 0] = 1  # (batch x max_occ)
        for increment in contribution:
            mass_movement = (count_prediction * increment.unsqueeze(1))[:, :max_occurence - 1]
            move = - torch.cat([mass_movement,
                                torch.cuda.FloatTensor(count_prediction.size()[0], 1).fill_(0)], axis=1) \
                   + torch.cat(
                [torch.cuda.FloatTensor(count_prediction.size()[0], 1).fill_(0),
                 mass_movement], axis=1)

            count_prediction = count_prediction + move

        return count_prediction

    def train_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:

        batch = self.train_batching(batch)

        preds = self.forward(batch, mode='train', epoch=engine.state.epoch)

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        self.backward(loss, engine.state.epoch)

        return self.prepare_output(preds, loss, batch)

    @torch.no_grad()
    def evaluate_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval', epoch=engine.state.epoch)

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        return self.prepare_output(preds, loss, batch)
