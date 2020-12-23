# ASC21-ClozeTest
## Project description
The project of ASC20-21 ClozeTest

## Requires
- python>=3.6.9
- pytorch>=1.6.0
- transformers>=4.0.1
- nltk

## Pretrained Model Prepare
This project currently support only BERT, Albert and Roberta pretrained model.<br>
To use one of model version supported, please pre-download from [hugging face model download page](https://huggingface.co/models).<br>
We heighly recommend you try to use **bert-large-uncased**, **albert-xlarge-v2** and **roberta-large** because they've been testd and proved worked.<br>

## Usage
### Recommand order to use
we recommand you to use this project in the following order:<br>
'eda_tools.py-->preprocess.py-->train.py'

### train.py
#### description
run this program to automatically preprocess and load data, train and evaluate model, and save model.
#### how to run
Run this on your command line like this:<br>
 ` CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --pretrained_model='albert-xlarge-v2'`
#### training tips
Each model has its best practice super parameters in this task.<br>
We've already tested some super parameters and they are showned below:<br>
pretrained-model|batch size|epoch|encoder lr|decoder lr
----------------|----------|-----|----------|----------
bert-large-uncased|8,16,32|5|1e-5,2e-5,3e-5,5e-5|encorder lr or 1e-4
albert-xlarge-v2|16,32,64|5|1e-5,2e-5,3e-5|encoder lr or 1e-4

*note that the batch size in above table is batch_size\*accumulate_steps in program.*
#### command line parameters
**--task_name**: type=str, default='ASC21ELE', help='task name of the training, do not change'<br>
**--pretrained_model**: type=str, default='bert-large-uncased', help='choose between bert-large-uncased, albert-xlarge-v2 and roberta-large'<br>
**--encoder_lr**: type=float, default=1e-5, help='learning rate for encoder'<br>
**--decoder_lr**: type=float, default=1e-4, help='learning rate for decoder, usually 10 times of the encoder_lr'<br>
**--accumulate_steps**: type=int, default=32, help='accumulation steps when training'<br>
**--max_epoch**: type=int, default=3, help='max epoch'<br>
**--batch_size**: type=int, default=2, help='batch size of data'<br>
**--fp16**: type=int, default=0, help='whether to use half precision, 1 to use and 0 not to'<br>
**--local_rank**: type=int, default=-1, help='local rank of device, do not change'<br>
**--debug**: type=int, default=0, help='1 to set the debug mode to run a fast test if there is any problem in the program, 0 to disable debug'<br>
**--use_baseline**: type=int, default=1, help='1 to use baseline model and 0 to use advanced model, in this version we only use baseline model'<br>
**--use_eda_data**: type=int, default=0, help='1 to use extra eda data, 0 to not'<br>

### preprocess.py
#### description
run this program to preprocess raw data into torch tensors.
#### how to run
run this on your command line like this:<br>
`python preprocess.py --pretrained_model=bert-large-uncased --ncpu=8`
#### command line parameters
**--pretrained_model**: pretrained_model_name,choose between *albert-xlarge-v2*,*bert-large-uncased*,*roberta-large*<br>
**--ncpu**: number of cpu core, default 0 that the program would automatically detect the value.<br>

### eda_tools.py
#### description
run this program to generate eda data whitch would improve the model performance
#### how to run
run this program on your command line like this:<br>
`python [path_to_your_root_dir]/tools/eda_tools.py`
