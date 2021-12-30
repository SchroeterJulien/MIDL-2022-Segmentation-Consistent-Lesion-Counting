# Segmentation-Consistent Lesion Counting Experiment Code from PVG-Pipeline

## Installing requirements
The pipeline requires at least python 3.8 to be installed. <br>
Install requirements using `pip install --no-cache -r requirements.txt`

## Run Instructions
### 1. Overview
- The experiment description files are located in `runner/experiments/gad`. This is where edits should be made to specific experiment runs (e.g. changes to hyperparameters).
- Experiment workflow and logic common to all experiments is in `runner/experiments/gad/template/template.py` but can be overrode in the experiment files
- Model architectures can be found in `runner/models/`
- "PB_Count" refers to our proposed model (Poisson Binomial Counting)
- The counting models require the segmentation model to be trained first (it will save the model at 20 epochs to be used as a pretrained model for the counting methods)
- To use comet, set an environment variable named "COMET_API_KEY" with your comet api key, otherwise be sure to disable comet logging. Setup the comet project in `runner/experiments/experiment_setup`
### 2. Dataset
Since the gadolinium-enhancing MS lesion dataset is private, the dataset component of the code is left blank (and marked by TODOs in train.py and evaluate.py).
The user must provide a pytorch dataset for their own dataset which returns a dictionary of the form:
<br>
{<br>
&emsp;&emsp;&emsp;&emsp; 'MRI': tensor of shape (B, C, D, H, W) -- the input to the network, <br>
&emsp;&emsp;&emsp;&emsp; 'LABEL': tensor of shape (B, C, D, H, W) -- the segmentation label, <br>
&emsp;&emsp;&emsp;&emsp; 'MASK': tensor of shape (B, C, D, H, W) -- the brain mask, <br>
&emsp;&emsp;&emsp;&emsp; 'COUNT': tensor of shape (B, 1) -- the ground truth lesion count, <br>
}<br>
where B refers to the batch size, C refers to the number of channels (e.g. MRI sequences), and D, H, W refer to the volume dimensions. The volumes must be resized to be of equal size (which can be done using
the resize flag in the Hyperparameters class if the batch size is exactly 1, otherwise
it must be done in the dataloader itself). 
### 3. Experiment Files
Experiment files contain all the hyperparameters specific to experiments. 
- Models, criterions, optimizers and schedulers can be spcified for each module in an architecture (e.g. unet vs. counting head)
- cp_ variables specify when and how often to save model checkpoints. See `runner/workflow/checkpointer.py`.
- Values that should be kept constant for all experiments can be added to `runner/experiments/gad/experiment_setup.py` and referenced
in experiment Hyperparameter classes. For example, comet project information or checkpointer values could be added to ensure
consistency across models.
### 4. Train
Run the experiment by filling in the variables in capital letters with the appropriate values (further instructions in runner/launch/train.py):
<br>
`python runner/launch/train.py -e EXPERIMENT_FOLDER -g GPU --clean`
<br>
For example, to run a segmentation-only experiment on gpu 0 without comet enabled, use:
<br>
`python runner/launch/train.py -e runner/experiments/gad/baseline_nnunet/segmentation_experiment -g 0 --disable-comet --clean`
<br>
The script will save all training metrics in a subfolder within the specified experiment directory. Metrics are saved in
`.csv` format. If comet is enabled, 
the training and validation curves will also be visible on the user's comet. 
### 5. Evaluate
The evaluate script works similarly to the train script. Use `-v` to use the validation set instead of the test set.
<br>
`python runner/launch/evaluate.py -e runner/experiments/gad/baseline_nnunet/segmentation_experiment -g 0 --disable-comet`
The script will automatically use the `best` model however, this can be changed by altering line 138 of `evaluate.py` to
load any saved model.

