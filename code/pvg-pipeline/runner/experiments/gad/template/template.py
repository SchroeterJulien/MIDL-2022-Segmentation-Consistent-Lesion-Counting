# Type checks
from typing import Callable, Dict, List, Tuple, Union, Optional
from ignite.metrics import Metric
from ignite.engine import Engine, Events
import torch
from torch import device as Device
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

# System imports
import sys
import os

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# Required project imports
from runner.builder.assembly import MetricAssembler, HandlerAssembler

# required imports
import torch.nn as nn


# user imports
from torch.optim import Adam
from ignite.metrics import Average, Accuracy
from runner.metrics.metric import BinaryThresholdConfusionMatrix, ApproximateMetrics, btcmF1,\
    WeightedBCEDecay
from runner.transforms.augmentation import CustomTransforms
from runner.models.nnunet import NNUNet, MultiNNUNet, SplitNNUNet

""" FUNCTIONAL """


def init_weights(m):
    if (type(m) == nn.Conv3d or
        type(m) == nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

""" EXPERIMENT """


class ExperimentTemplate:
    def __init__(self, Hyperparameters):
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.criterions = {}
        self.HP = Hyperparameters
        self.train_metrics: Optional[Tuple[List[Metric], List[str]]] = None
        self.evaluate_metrics: Optional[Tuple[List[Metric], List[str]]] = None
        self.test_metrics: Optional[Tuple[List[Metric], List[str]]] = None
        self.train_handlers: Optional[Tuple[List[Callable], List[Events]]] = None
        self.evaluate_handlers: Optional[Tuple[List[Callable], List[Events]]] = None
        self.test_handlers: Optional[Tuple[List[Callable], List[Events]]] = None
        self.devices: Optional[List[Device]] = None
        self.augmentations = []

    ############### IMPLEMENT FUNCTIONS BELOW #########################
    def setup_devices(self, devices:Optional[List[Device]]=None)->None:
        if devices is None:
            self.devices = ['cpu']
            # raise Exception("No device received.")
            # Alt option: define CPU device?
        else:
            self.devices = devices

    def setup_models(self)->None:
        unet = NNUNet(**self.HP.models['unet'])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet

    def setup_optimizers(self)->None:
        self.optimizers['unet_adam'] = Adam(self.models['unet'].parameters(), **self.HP.optimizers['unet_adam'])

    def setup_criterions(self)->None:
        self.criterions['unet_loss'] = self.setup_bce(net='unet_loss')

    def setup_schedulers(self)->None:
        if len(self.HP.schedulers) > 0:
            self.schedulers['unet_scheduler'] = ExponentialLR(self.optimizers['unet_adam'],
                                                              **self.HP.schedulers['unet_scheduler'])

    def setup_objects(self)->None:
        # Models
        self.setup_models()

        # Optimizers
        self.setup_optimizers()

        # Schedulers
        self.setup_schedulers()

        # Criterions
        self.setup_criterions()

        try:
            # Data augmentation
            input_keys = ['MRI', 'LABEL', 'MASK']
            if self.HP.data_augmentations['WrappedRandomAffine3D'] is not None:
                input_mapping = {'MRI': [1, False], 'LABEL': [0, False],
                                 'MASK': [0, False]}
                self.augmentations.append(CustomTransforms.WrappedRandomAffine3D(input_mapping=input_mapping,
                                                                                 **self.HP.data_augmentations[
                                                                                     'WrappedRandomAffine3D'],
                                                                                 device=self.devices[0]))
            if self.HP.data_augmentations['WrappedRandomHorizontalFlip'] is not None:
                self.augmentations.append(CustomTransforms.WrappedRandomHorizontalFlip(input_mapping=input_keys,
                                                                                       **self.HP.data_augmentations[
                                                                                           'WrappedRandomHorizontalFlip'],
                                                                                       device=self.devices[0]))
            if self.HP.data_augmentations['WrappedRandomVerticalFlip'] is not None:
                self.augmentations.append(CustomTransforms.WrappedRandomVerticalFlip(input_mapping=input_keys,
                                                                                     **self.HP.data_augmentations[
                                                                                         'WrappedRandomVerticalFlip'],
                                                                                     device=self.devices[0]))
            if 'WrappedRandomDepthicalFlip' in self.HP.data_augmentations:
                if self.HP.data_augmentations['WrappedRandomDepthicalFlip'] is not None:
                    self.augmentations.append(CustomTransforms.WrappedRandomDepthicalFlip(input_mapping=input_keys,
                                                                                           **self.HP.data_augmentations[
                                                                                               'WrappedRandomDepthicalFlip'],
                                                                                           device=self.devices[0]))
        except KeyError or AttributeError:
            print('No data augmentations selected')
            return

    def setup_bce(self, net, reduction='none'):

        if self.HP.criterions[net]['weighted']:
            if self.HP.criterions[net]['decaying']:
                return WeightedBCEDecay(pos_weight=self.HP.criterions[net]['pos_weight'],
                                        decay_factor=self.HP.criterions[net]['decay_factor'],
                                        reduction=reduction)

            else:
                return nn.BCEWithLogitsLoss(reduction=reduction,
                                            pos_weight=torch.tensor(self.HP.criterions['unet_loss']['pos_weight']))
        else:
            return nn.BCEWithLogitsLoss(reduction=reduction)

    def setup_train_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'train_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds', label_key='label', device=self.devices[0] )
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'train_PR_AUC')
        assembler.add(ROC_AUC, 'train_ROC_AUC')
        assembler.add(F1, 'train_F1')

        self.train_metrics = assembler.build()

    def setup_evaluate_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'val_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds', label_key='label', device=self.devices[0] )
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'val_PR_AUC')
        assembler.add(ROC_AUC, 'val_ROC_AUC')
        assembler.add(F1, 'val_F1')

        self.evaluate_metrics = assembler.build()

    def setup_test_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'test_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds', label_key='label', device=self.devices[0] )
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'test_PR_AUC')
        assembler.add(ROC_AUC, 'test_ROC_AUC')
        assembler.add(F1, 'test_F1')

        self.evaluate_metrics = assembler.build()

    def setup_train_handlers(self) -> None:
        if not len(self.HP.schedulers) > 0:
            return

        # create generic step method
        def scheduler_step(engine):
            self.schedulers['unet_scheduler'].step()

        assembler = HandlerAssembler()
        assembler.add(scheduler_step, Events.EPOCH_COMPLETED)
        self.train_handlers = assembler.build()

    def setup_evaluate_handlers(self)->None:
        pass

    def setup_test_handlers(self)->None:
        pass

    def train_batching(self, batch: Union[Dict, Tensor])->Union[Dict, List, Tensor, Tuple]:
        del batch['COUNT']

        # Cast labels to float
        batch['LABEL'] = batch['LABEL'].float()

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        # Pad the input to be an even size
        if self.HP.resize:
            batch = CustomTransforms.resize(batch, size=self.HP.size, imaging_keys=list(batch.keys()))

        # Augment
        for augmentation in self.augmentations:
            batch = augmentation(batch)

        return batch

    def evaluate_batching(self, batch: Union[Dict, Tensor])->Union[Dict, List, Tensor, Tuple]:
        del batch['COUNT']

        # Cast labels to float
        batch['LABEL'] = batch['LABEL'].float()

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        if self.HP.resize:
            batch = CustomTransforms.resize(batch, size=self.HP.size, imaging_keys=list(batch.keys()))

        return batch

    def forward(self, batch, mode='train'):
        model = self.models['unet']
        if mode == 'train':
            model.train()
        else:
            model.eval()
        preds = model(batch['MRI'])
        return preds

    def calculate_losses(self, preds, batch, epoch=None):
        if self.HP.criterions['unet_loss']['decaying']:
            loss = self.criterions['unet_loss'](preds, batch['LABEL'], epoch)
        else:
            loss = self.criterions['unet_loss'](preds, batch['LABEL'])
        loss = loss[batch['MASK'].bool()].mean()
        return loss

    def backward(self, loss, epoch=None):
        optimizer = self.optimizers['unet_adam']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def prepare_output(self, preds, loss, batch):
        return {'preds': preds[batch['MASK'].bool()].detach(),
                'label': batch['LABEL'][batch['MASK'].bool()].detach(),
                'loss': loss.item()
                }

    def train_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:
        batch = self.train_batching(batch)

        preds = self.forward(batch, mode='train')

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        self.backward(loss, engine.state.epoch)

        return self.prepare_output(preds, loss, batch)

    @torch.no_grad()
    def evaluate_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval')

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        return self.prepare_output(preds, loss, batch)

    @torch.no_grad()
    def test_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval')

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        return self.prepare_output(preds, loss, batch)

    ##################################################################

    def get_models(self):
        return self.models

    def get_optimizers(self):
        return self.optimizers

    def get_schedulers(self):
        return self.schedulers

    def get_criterions(self):
        return self.criterions

    def get_train_metrics(self):
        return self.train_metrics

    def get_evaluate_metrics(self):
        return self.evaluate_metrics

    def get_test_metrics(self):
        return self.test_metrics

    def get_train_handlers(self):
        return self.train_handlers

    def get_evaluate_handlers(self):
        return self.evaluate_handlers

    def get_test_handlers(self):
        return self.test_handlers


class ExperimentCountTemplate(ExperimentTemplate):
    def __init__(self, Hyperparameters):
        super(ExperimentCountTemplate, self).__init__(Hyperparameters)
        if self.HP.bins is not None:
            self.bin = CustomTransforms.BinCount(bins=self.HP.bins)
        else:
            self.bin = None
        self.i = 0

    ############### IMPLEMENT FUNCTIONS BELOW #########################

    def setup_train_metrics(self)->None:
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
        acc = Accuracy(output_transform=lambda output: (output['count_pred'], output['count']), device=self.devices[0])
        assembler.add(acc, 'train_acc')

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

        acc = Accuracy(output_transform=lambda output: (output['count_pred'], output['count']), device=self.devices[0])
        assembler.add(acc, 'val_acc')

        self.evaluate_metrics = assembler.build()

    def setup_test_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['seg_loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'test_seg_avg_loss')

        avg_count_loss_metric = Average(output_transform=lambda output: output['count_loss'], device=self.devices[0])
        assembler.add(avg_count_loss_metric, 'test_count_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds',
                                              label_key='label', device=self.devices[0])
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'test_PR_AUC')
        assembler.add(ROC_AUC, 'test_ROC_AUC')
        assembler.add(F1, 'test_F1')

        acc = Accuracy(output_transform=lambda output: (output['count_pred'], output['count']), device=self.devices[0])
        assembler.add(acc, 'test_acc')

        self.test_metrics = assembler.build()

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
            batch = CustomTransforms.resize(batch, size=self.HP.size,
                                            imaging_keys=['MRI', 'LABEL', 'MASK'])

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
            batch = CustomTransforms.resize(batch,size=self.HP.size, imaging_keys=['MRI', 'LABEL', 'MASK'])

        return batch


class ExperimentBottleneckTemplate(ExperimentCountTemplate):
    def __init__(self, Hyperparameters):
        super(ExperimentBottleneckTemplate, self).__init__(Hyperparameters)

    ############### IMPLEMENT FUNCTIONS BELOW #########################

    def setup_models(self, Bottleneck)->None:
        unet = MultiNNUNet(**self.HP.models['unet'])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet

        bottleneck = Bottleneck(self.HP, **self.HP.models['bottleneck'])
        bottleneck = bottleneck.to(self.devices[0])
        bottleneck.apply(init_weights)
        self.models['bottleneck'] = bottleneck

    def setup_optimizers(self) ->None:
        super().setup_optimizers()
        self.optimizers['bottleneck_adam'] = Adam(self.models['bottleneck'].parameters(), **self.HP.optimizers['bottleneck_adam'])

    def setup_criterions(self) ->None:
        super().setup_criterions()
        if self.HP.criterions['bottleneck_loss']['weighted']:
            self.criterions['bottleneck_loss'] = nn.CrossEntropyLoss(reduction='mean',
                                                                    weight=torch.tensor(self.HP.criterions['bottleneck_loss']['pos_weight'],
                                                                                        device=self.devices[0]))
        else:
            self.criterions['bottleneck_loss'] = nn.CrossEntropyLoss(reduction='mean')

    def setup_schedulers(self) ->None:
        super().setup_schedulers()
        if len(self.HP.schedulers) > 0:
            self.schedulers['bottleneck_scheduler'] = ExponentialLR(self.optimizers['bottleneck_adam'],
                                                           **self.HP.schedulers['bottleneck_scheduler'])

    def setup_train_handlers(self) -> None:
        if not len(self.HP.schedulers) > 0:
            return

        # create generic step method
        def scheduler_step(engine):
            self.schedulers['bottleneck_scheduler'].step()
            self.schedulers['unet_scheduler'].step()

        assembler = HandlerAssembler()
        assembler.add(scheduler_step, Events.EPOCH_COMPLETED)
        self.train_handlers = assembler.build()

    def forward(self, batch, mode='train'):
        unet = self.models['unet']
        bottleneck = self.models['bottleneck']
        if mode == 'train':
            unet.train()
            bottleneck.train()
        else:
            unet.eval()
            bottleneck.eval()
        # in this experiment, use the segmentation label as the input
        seg_pred, bottleneck_in = unet(batch['MRI'])
        count_pred = bottleneck(bottleneck_in)
        preds = (seg_pred, count_pred)

        return preds

    def calculate_losses(self, preds, batch, epoch=None):
        seg_loss = super().calculate_losses(preds[0], batch, epoch)
        count_loss = self.criterions['bottleneckloss'](preds[1], torch.argmax(batch['COUNT'], dim=1))
        return (seg_loss, count_loss)

    def backward(self, loss, epoch=None):
        unet_optimizer = self.optimizers['unet_adam']
        bottleneck_optimizer = self.optimizers['bottleneck_adam']
        unet_optimizer.zero_grad()
        bottleneck_optimizer.zero_grad()

        weight = self.HP.w1 * torch.exp(self.HP.w2 * epoch)
        total_loss = loss[0] + weight * loss[1]

        total_loss.backward()
        unet_optimizer.step()
        bottleneck_optimizer.step()

    def prepare_output(self, preds, loss, batch):
        count_pred = preds[1]
        one_hot = torch.zeros(size=(count_pred.size(0), count_pred.size(1)), device=self.devices[0])
        one_hot[torch.arange(count_pred.size(0)), torch.argmax(count_pred, dim=1)] = 1.0
        seg_pred = preds[0]
        return {'preds': seg_pred[batch['MASK'].bool()].detach(),
                'label': batch['LABEL'][batch['MASK'].bool()].detach(),
                'count': torch.argmax(batch['COUNT'], dim=1).detach(),
                'count_pred': one_hot.detach(),
                'seg_loss': loss[0].item(),
                'count_loss': loss[1].item()
                }


class ExperimentDownstreamTemplate(ExperimentCountTemplate):
    def __init__(self, Hyperparameters):
        super(ExperimentDownstreamTemplate, self).__init__(Hyperparameters)

    ############### IMPLEMENT FUNCTIONS BELOW #########################

    def setup_models(self, Downstream)->None:
        downstream = Downstream(self.HP, **self.HP.models['downstream'])
        downstream = downstream.to(self.devices[0])
        downstream.apply(init_weights)
        self.models['downstream'] = downstream

    def setup_optimizers(self) ->None:
        self.optimizers['downstream_adam'] = Adam(self.models['downstream'].parameters(), **self.HP.optimizers['downstream_adam'])
        super().setup_optimizers(self)

    def setup_criterions(self) ->None:
        if self.HP.criterions['downstream_loss']['weighted']:
            self.criterions['downstream_loss'] = nn.CrossEntropyLoss(reduction='mean',
                                                                    weight=torch.tensor(self.HP.criterions['downstream_loss']['pos_weight'],
                                                                                        device=self.devices[0]))
        else:
            self.criterions['downstream_loss'] = nn.CrossEntropyLoss(reduction='mean')
        super().setup_criterions()

    def setup_schedulers(self) ->None:
        if len(self.HP.schedulers) > 0:
            self.schedulers['downstream_scheduler'] = ExponentialLR(self.optimizers['downstream_adam'],
                                                           **self.HP.schedulers['downstream_scheduler'])
        super().setup_schedulers()

    def setup_train_handlers(self) -> None:
        if not len(self.HP.schedulers) > 0:
            return

        # create generic step method
        def scheduler_step(engine):
            self.schedulers['downstream_scheduler'].step()
            self.schedulers['unet_scheduler'].step()

        assembler = HandlerAssembler()
        assembler.add(scheduler_step, Events.EPOCH_COMPLETED)
        self.train_handlers = assembler.build()

    def forward(self, batch, mode='train'):
        unet = self.models["unet"]
        model = self.models['downstream']
        if mode == 'train':
            model.train()
            unet.train()
        else:
            model.eval()
            unet.eval()
        seg = unet(batch["MRI"])
        seg_copy = seg.clone()
        seg_copy = torch.sigmoid(seg_copy)

        preds = model(seg_copy)

        return seg, preds

    def calculate_losses(self, preds, batch, epoch=None):
        count_loss = self.criterions['downstream_loss'](preds[1], torch.argmax(batch['COUNT'], dim=1))
        unet_loss = super().calculate_losses(self, preds[0], batch)
        return unet_loss, count_loss

    def backward(self, loss, epoch=None):
        unet_optimizer = self.optimizers['unet_adam']
        downstream_optimizer = self.optimizers['downstream_adam']
        unet_optimizer.zero_grad()
        downstream_optimizer.zero_grad()

        weight = self.HP.w1 * torch.exp(self.HP.w2 * epoch)
        total_loss = loss[0] + weight * loss[1]

        total_loss.backward()
        unet_optimizer.step()
        downstream_optimizer.step()

    def prepare_output(self, preds, loss, batch):
        count_pred = preds[1]
        one_hot = torch.zeros(size=(count_pred.size(0), count_pred.size(1)), device=self.devices[0])
        one_hot[torch.arange(count_pred.size(0)), torch.argmax(count_pred, dim=1)] = 1.0
        seg_pred = preds[0]
        return {'preds': seg_pred[batch['MASK'].bool()].detach(),
                'label': batch['LABEL'][batch['MASK'].bool()].detach(),
                'count': torch.argmax(batch['COUNT'], dim=1).detach(),
                'count_pred': one_hot.detach(),
                'seg_loss': loss[0].item(),
                'count_loss': loss[1].item()
                }


class ExperimentMultiHeadTemplate(ExperimentCountTemplate):
    def __init__(self, Hyperparameters):
        super(ExperimentMultiHeadTemplate, self).__init__(Hyperparameters)

    ############### IMPLEMENT FUNCTIONS BELOW #########################

    def setup_models(self, Multihead)->None:
        unet = SplitNNUNet(**self.HP.models['unet'])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet

        multihead = Multihead(self.HP, **self.HP.models['multihead'])
        multihead = multihead.to(self.devices[0])
        multihead.apply(init_weights)
        self.models['multihead'] = multihead

    def setup_optimizers(self) ->None:
        super().setup_optimizers()
        self.optimizers['multihead_adam'] = Adam(self.models['multihead'].parameters(), **self.HP.optimizers['multihead_adam'])

    def setup_criterions(self) ->None:
        super().setup_criterions()
        if self.HP.criterions['multihead_loss']['weighted']:
            self.criterions['multihead_loss'] = nn.CrossEntropyLoss(reduction='mean',
                                                                    weight=torch.tensor(self.HP.criterions['multihead_loss']['pos_weight'],
                                                                    device=self.devices[0]))
        else:
            self.criterions['multihead_loss'] = nn.CrossEntropyLoss(reduction='mean')

    def setup_schedulers(self) ->None:
        super().setup_schedulers()
        if len(self.HP.schedulers) > 0:
            self.schedulers['multihead_scheduler'] = ExponentialLR(self.optimizers['multihead_adam'],
                                                           **self.HP.schedulers['multihead_scheduler'])

    def setup_train_handlers(self) -> None:
        if not len(self.HP.schedulers) > 0:
            return

        # create generic step method
        def scheduler_step(engine):
            self.schedulers['multihead_scheduler'].step()
            self.schedulers['unet_scheduler'].step()

        assembler = HandlerAssembler()
        assembler.add(scheduler_step, Events.EPOCH_COMPLETED)
        self.train_handlers = assembler.build()

    def forward(self, batch, mode='train'):
        unet = self.models['unet']
        multihead = self.models['multihead']
        if mode == 'train':
            unet.train()
            multihead.train()
        else:
            unet.eval()
            multihead.eval()
        seg_pred, output = unet(batch['MRI'])
        count_pred = multihead(output)
        preds = (seg_pred, count_pred)
        return preds

    def calculate_losses(self, preds, batch, epoch=None):
        seg_loss, count_loss = super().calculate_losses(preds, batch, epoch)

        weight = self.HP.w1 * torch.exp(self.HP.w2 * epoch)
        total_loss = seg_loss + weight * count_loss

        return (seg_loss, count_loss, total_loss)

    def backward(self, loss, epoch):
        unet_optimizer = self.optimizers['unet_adam']
        multihead_optimizer = self.optimizers['multihead_adam']
        unet_optimizer.zero_grad()
        multihead_optimizer.zero_grad()

        loss[2].backward()
        unet_optimizer.step()
        multihead_optimizer.step()

    def prepare_output(self, preds, loss, batch):
        seg_pred = preds[0]
        preds = preds[1]
        one_hot = torch.zeros(size=(preds.size(0), preds.size(1)), device=self.devices[0])
        one_hot[torch.arange(preds.size(0)), torch.argmax(preds, dim=1)] = 1.0
        return {'preds': seg_pred[batch['MASK'].bool()].detach(),
                'label': batch['LABEL'][batch['MASK'].bool()].detach(),
                'count': torch.argmax(batch['COUNT'], dim=1).detach(),
                'count_pred': one_hot.detach(),
                'seg_loss': loss[0].item(),
                'count_loss': loss[1].item()
                }
