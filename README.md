# ASC21-ClozeTest
## Project description
The project of ASC20-21 ClozeTest
## Requires
- python>=3.6.9
- pytorch>=1.6.0
- transformers>=4.0.1
## Usage
### train.py
to train a model,using the train.py file.

run this on your command line like this:
 ` CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --pretrained_model='albert-xlarge-v2'`
