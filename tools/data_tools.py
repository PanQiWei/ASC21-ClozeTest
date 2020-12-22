import torch
from torch.utils.data import Dataset, DataLoader

from preprocess import Sample
from config import *

import random

class ELEDataset(Dataset):
    def __init__(self, pt_path, debug):
        self.samples = torch.load(pt_path)
        self.samples.sort(key= lambda x: len(x.article), reverse=True)
        if debug:
            self.samples = self.samples[:100]
        self.max_article_len = 512
        self.max_option_len = 0
        self.max_option_num = 0
        for idx in range(len(self.samples)):
            sample = self.samples[idx]
            for ops in sample.ops:
                for op in ops:
                    self.max_option_len = max(self.max_option_len, op.size(0))
            self.max_option_num  = max(self.max_option_num, len(sample.ops))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # get sample name and rename
        sample_name = sample.sample_name
        if 'test' in sample_name:
            order, part = sample_name[4:].split("-")
            sample_name = int('3'+order+part)
        else:
            sample_name = 0
        # initialize some tensors needed
        article=torch.zeros(self.max_article_len).long()
        article_mask=torch.zeros(self.max_article_len)
        option=torch.zeros(self.max_option_num,4,
                           self.max_option_len).long()
        option_mask=torch.zeros(self.max_option_num,4,
                           self.max_option_len)
        answer=torch.zeros(self.max_option_num).long()
        blank_pos=torch.zeros(self.max_option_num).long()
        mask=torch.zeros(self.max_option_num)
        
        # produce tensors created above
        article[:sample.article.size(0)] = sample.article
        article_mask[:sample.article.size(0)]= 1
        if len(sample.ans) > 0:
            answer[:sample.ans.size(0)] = sample.ans
        blank_pos[:sample.blank.size(0)] = sample.blank
        mask[:sample.blank.size(0)] = 1
        for i,ops in enumerate(sample.ops):
            for j, op in enumerate(ops):
                option[i, j, :op.size(0)] = op
                option_mask[i, j, :op.size(0)] = 1
                
        return (article, option, answer, article_mask, option_mask, mask, blank_pos, sample_name)
