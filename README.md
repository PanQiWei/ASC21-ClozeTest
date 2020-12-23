# ASC21-ClozeTest
## Project description
The project of ASC20-21 ClozeTest
## Requires
- python>=3.6.9
- pytorch>=1.6.0
- transformers>=4.0.1
## Pretrained Model Prepare
This project currently support only BERT, Albert and Roberta pretrained model.<br>
To use one of model version supported, please pre-download from [hugging face model download page](https://huggingface.co/models).<br>
We heighly recommend you try to use **bert-large-uncased**, **albert-xlarge-v2** and **roberta-large** because they've been testd and proved worked.<br>
## Usage
### train.py
#### how to ran
To train a model,using the train.py file.<br>
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
