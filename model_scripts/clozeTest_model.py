from config import *

from transformers import AutoModel
from transformers.activations import ACT2FN

import torch
import torch.nn as nn
import torch.nn.functional as F

PRETRAINED_MODEL_ROOT = ROOT+"/pretrained_models/"

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

class PredictionHead(nn.Module):
	def __init__(self, config, pretrained_model_name):
		super().__init__()
		if 'roberta' in pretrained_model_name:
			self.model_name = 'roberta'
		elif 'albert' in pretrained_model_name:
			self.model_name = 'albert'
		elif 'bert' in pretrained_model_name:
			self.model_name = 'bert'
		else:
			raise ValueError("the model not supported yet.")

		if self.model_name == 'albert':
			self.layer_norm = nn.LayerNorm(config.embedding_size)
			self.bias = nn.Parameter(torch.zeros(config.vocab_size))
			self.dense = nn.Linear(config.hidden_size, config.embedding_size)
			self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
			self.activation = ACT2FN[config.hidden_act]
		elif self.model_name == 'roberta' or self.model_name == 'bert':
			self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
			self.bias = nn.Parameter(torch.zeros(config.vocab_size))
			self.dense = self.dense = nn.Linear(config.hidden_size, config.hidden_size)
			self.decoder = self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
			self.activation = F.gelu

		self.decoder.bias = self.bias

	def init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.activation(hidden_states)
		hidden_states = self.layer_norm(hidden_states)
		hidden_states = self.decoder(hidden_states)

		prediction_scores = hidden_states

		return prediction_scores

class ClozeTestModel(nn.Module):
	def __init__(self, pretrained_model_name, debug):
		super(ClozeTestModel, self).__init__()
		self.pretrained_model = AutoModel.from_pretrained(PRETRAINED_MODEL_ROOT+pretrained_model_name)
		self.prediction_head = PredictionHead(self.pretrained_model.config, pretrained_model_name)
		self.debug =debug

		# init weights
		self.prediction_head.apply(self.prediction_head.init_weights)
	def forward(self, batch):
		article, option, answer, article_mask, option_mask, mask, blank_pos, _ = batch
		article, article_mask = repadding(article, article_mask)

		batch_size = option.size(0)
		option_num = option.size(1)

		last_hidden_state = self.pretrained_model(article, article_mask)[0] # shape:(batch_size, sequence_length, hidden_size)
		# get masked words' last hidden state from last_hidden_state
		blank_pos = blank_pos.unsqueeze(-1)
		blank_pos = blank_pos.expand(batch_size, option_num, last_hidden_state.size(-1)) # reshape blank_pos as shape of [batch_size, option_num(20), hidden_size]
		masked_word_last_hidden_state = torch.gather(last_hidden_state, 1, blank_pos) # gather last-hidden-state of masked words, shape:[batch_size,,option_num(20), hidden_size]
		# using prediction head to predict the score of each word in vocabulary that the masked word is.
		masked_word_logits = self.prediction_head(masked_word_last_hidden_state)
		masked_word_logits = masked_word_logits.view(batch_size, option_num, 1, self.pretrained_model.config.vocab_size)
		masked_word_logits = masked_word_logits.expand(batch_size, option_num, 4, self.pretrained_model.config.vocab_size)
		# get candidate words score from masked_word_logits.
		masked_word_logits_only_candidates = torch.gather(masked_word_logits, -1, option) # prediction scores of each candidtes of each masked word. shape:[batch_size, option_num(20), 4, max_option_ids_length]
		masked_word_logits_only_candidates = masked_word_logits_only_candidates*option_mask
		masked_word_logits_only_candidates = masked_word_logits_only_candidates.sum(-1).float() # shape:[batch_size, option_num(20), 4]
		masked_word_logits_only_candidates = F.softmax(masked_word_logits_only_candidates, dim=-1)

		out = masked_word_logits_only_candidates.view(-1,4)

		return batch_size, option_num, out