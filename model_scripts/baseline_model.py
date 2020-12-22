from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForMaskedLM

CACHE_DIR = ROOT+"/pretrained_models/"

def repadding(article, article_mask):
	"""
    re-padding article tensor in order to speed up training
    """
	batch_max_article_len = 0
	batch_size = article.size(0)
	for i in range(batch_size):
		each_article = article[i]
		batch_max_article_len = max((each_article != 0).sum(), batch_max_article_len)

	new_article = torch.zeros([article.size(0), batch_max_article_len]).long().type_as(article)
	new_article_mask = torch.zeros([article.size(0), batch_max_article_len]).long().type_as(article_mask)
	for i in range(batch_size):
		new_article[i] = article[i][:batch_max_article_len]
		new_article_mask[i] = article_mask[i][:batch_max_article_len]

	return new_article, new_article_mask

class BaseLineModel(nn.Module):
    def __init__(self, pretrained_model_name, debug):
        super(BaseLineModel, self).__init__()
        self.pretrained_model = AutoModelForMaskedLM.from_pretrained(CACHE_DIR+pretrained_model_name)
        self.debug = debug

    def freeze(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, batch, mode='train'):
        article, option, answer, article_mask, option_mask, mask, blank_pos, _ = batch
        article, article_mask = repadding(article, article_mask)

        batch_size = option.size(0)
        option_num = option.size(1)

        outs = self.pretrained_model(article, article_mask)
        logits = outs.logits # score of each word in vocab. shape:[batch_size, sequence_length, vocab_size]
        blank_pos = blank_pos.unsqueeze(-1)
        blank_pos = blank_pos.expand(batch_size, option_num, logits.size(-1)) # reshape blank_pos as shape of [batch_size, option_num(20), vocab_size]
        masked_word_logits = torch.gather(logits, 1, blank_pos) # get the prediction scores where the MASK tokens are. shape:[batch_size, option_num(20), vocab_size]
        masked_word_logits = masked_word_logits.view(batch_size, option_num, 1, self.pretrained_model.config.vocab_size)
        masked_word_logits = masked_word_logits.expand(batch_size, option_num, 4, self.pretrained_model.config.vocab_size)
        masked_word_logits_only_candidates = torch.gather(masked_word_logits, -1, option) # prediction scores of each candidtes of each masked word. shape:[batch_size, option_num(20), 4, max_option_ids_length]
        masked_word_logits_only_candidates = masked_word_logits_only_candidates*option_mask
        masked_word_logits_only_candidates = masked_word_logits_only_candidates.sum(-1).float() # shape:[batch_size, option_num(20), 4]
        masked_word_logits_only_candidates = F.softmax(masked_word_logits_only_candidates, dim=-1)
        out = masked_word_logits_only_candidates.view(-1,4)

        return batch_size, option_num, out