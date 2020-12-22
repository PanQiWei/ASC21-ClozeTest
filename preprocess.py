import os, sys
from config import *
sys.path.append(ROOT+"/tools/")
import path_tools

import glob
import time

import numpy as np
import torch
import json
import argparse
import random
import multiprocessing as mul
from multiprocessing import Manager, Process, Pool, Queue, Lock

try:
    import transformers
except:
    os.system("pip install transformers")
finally:
    from transformers import BertTokenizer, AlbertTokenizer,RobertaTokenizer,XLNetTokenizer

torch.multiprocessing.set_sharing_strategy('file_system')

class Sample(object):
    def __init__(self, sample_name):
        self.sample_name = sample_name # sample name
        self.article = None # article
        self.blank = [] # idx of blank in article
        self.ops = [] # options
        self.ans = [] # answers
                    
    def convert_tokens_to_ids(self, tokenizer):
        self.article = tokenizer.convert_tokens_to_ids(self.article)
        self.article = torch.Tensor(self.article)
        for i in range(len(self.ops)):
            for k in range(4):
                self.ops[i][k] = tokenizer.convert_tokens_to_ids(self.ops[i][k])
                self.ops[i][k] = torch.Tensor(self.ops[i][k])
        self.blank = torch.Tensor(self.blank)
        if len(self.ans) > 0:
            self.ans = torch.Tensor(self.ans)

class Preprocessor(object):
    def __init__(self, args):
        '''
        args contains: pretrained_model, ncpu, data_dir, out_path
        '''
        PRETRAINED_MODEL_DIR = ROOT+"/pretrained_models/"
        PRETRAINED_MODEL = {
            "bert-large-uncased":PRETRAINED_MODEL_DIR+"bert-large-uncased",
            "albert-base-v1":PRETRAINED_MODEL_DIR+"albert-base-v1",
            "albert-large-v1":PRETRAINED_MODEL_DIR+"albert-large-v1",
            "albert-base-v2":PRETRAINED_MODEL_DIR+"albert-base-v2",
            "albert-large-v2":PRETRAINED_MODEL_DIR+"albert-large-v2",
            "albert-xlarge-v2":PRETRAINED_MODEL_DIR+"albert-xlarge-v2",
            "albert-xxlarge-v2":PRETRAINED_MODEL_DIR+"albert-xxlarge-v2",
            "roberta-large":PRETRAINED_MODEL_DIR+"roberta-large",
            "xlnet-large-cased":PRETRAINED_MODEL_DIR+"xlnet-large-cased"
        }
        self.tokenizer = None
        self.mask_token = None
        if 'albert' in args.pretrained_model:
            self.tokenizer = AlbertTokenizer.from_pretrained(PRETRAINED_MODEL[args.pretrained_model])
            self.mask_token = '[MASK]'
        elif 'roberta' in args.pretrained_model:
            self.tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL[args.pretrained_model])
            self.mask_token = '<mask>'
        elif 'xlnet' in args.pretrained_model:
            self.tokenizer = XLNetTokenizer.from_pretrained(PRETRAINED_MODEL[args.pretrained_model])
            self.mask_token = '<mask>'
        elif 'bert' in args.pretrained_model:
            self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL[args.pretrained_model])
            self.mask_token = '[MASK]'
        else:
            raise ValueError("{} model dosen't support yet.".format(args.pretrained_model))

        self.data_dir = args.data_dir # directory contains raw data
        self.out_path = args.out_path # path of .pt file which saves data preprocessed
        self.files = path_tools.get_files(self.data_dir, extension='json') # all raw files
        self.ncpu = max(1,args.ncpu-2)

    def preprocess(self):
        print("main process: preprocessing raw data from {}...".format(self.data_dir))
        group_len = len(self.files)//self.ncpu
        files_groups = [self.files[i*group_len:(i+1)*group_len] if i!=self.ncpu-1 else self.files[i*group_len:] for i in range(self.ncpu)]

        queue = Queue()
        lock = Lock()
        process_list = [Process(target=self._create_sample, args=(files_group, queue, lock)) for files_group in files_groups]
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
        
        samples = []
        print("main process: getting all samples")
        for i in range(self.ncpu):
            temp_path = queue.get()
            temp_samples = torch.load(temp_path)
            samples += temp_samples
            os.remove(temp_path)

        print("main process: preprocess complete.")
        return samples

    def save(self, samples):
        print("mainp process: saving samples to {}".format(self.out_path))
        torch.save(samples, self.out_path)
        print("main process: done.")


    def _create_sample(self, files, queue, lock):
        """
        create Sample object from raw data
        """
        print("process {}: ready for starting".format(os.getpid()))
        time.sleep(1)
        samples = []
        for file in files:
            cnt = 0 # counter used to match answer and options from question(blank in this case)
            raw_data = json.loads(open(file,'r').read()) # json data format
            _, tmp_fname = os.path.split(file)
            fname, _ = os.path.splitext(tmp_fname) # file name, later will be used as sample name

            #print("process {}: createing sample for {}".format(os.getpid() ,fname))

            # tokenze article
            article = raw_data['article'].replace('_', self.mask_token)
            article = self.tokenizer.tokenize(article)

            # separate article if needed
            if(len(article)<=512):
                sample1 = Sample(fname+'-0')
                sample1.article = article
                for idx in range(len(article)):
                    if (article[idx] == self.mask_token):
                        sample1.blank.append(idx)
                        ops = self._tokenize_ops(raw_data['options'][cnt], self.tokenizer)
                        sample1.ops.append(ops)
                        if raw_data['answers']:
                            sample1.ans.append(ord(raw_data['answers'][cnt]) - ord('A')) # answer A,B,C,D->0,1,2,3
                        cnt += 1
                sample1.convert_tokens_to_ids(self.tokenizer)
                samples += [sample1]
            else:
                sample1 = Sample(fname+'-0')
                sample2 = Sample(fname+'-1')

                for idx in range(len(article)):
                    if (article[idx] == self.mask_token):
                        ops = self._tokenize_ops(raw_data['options'][cnt], self.tokenizer)
                        if (idx < 512):
                            sample1.blank.append(idx)
                            sample1.ops.append(ops)
                            if raw_data['answers']:
                                sample1.ans.append(ord(raw_data['answers'][cnt]) - ord('A'))
                        else:
                            sample2.blank.append(idx - 512)
                            sample2.ops.append(ops)
                            if raw_data['answers']:
                                sample2.ans.append(ord(raw_data['answers'][cnt]) - ord('A'))
                        cnt += 1
                sample1.article = article[:512]
                sample2.article = article[-512:]

                sample1.convert_tokens_to_ids(self.tokenizer)
                # only saved sample2 when there are blanks in its article
                if (len(sample2.blank) == 0):
                    samples += [sample1]
                else:
                    sample2.convert_tokens_to_ids(self.tokenizer)
                    samples += [sample1, sample2]

        # save to temp file
        temp_path = self.out_path[:-3]+str(os.getpid())+'.pt'
        print("process {}: saving samples to temp file {}".format(os.getpid(), temp_path))
        torch.save(samples, temp_path)
        lock.acquire()
        queue.put(temp_path)
        lock.release()
        print("process {}: preprocess compelete.".format(os.getpid()))

    def _tokenize_ops(self, ops, tokenizer):
        ret = []
        for i in range(4):
            ret.append(tokenizer.tokenize(ops[i]))
        return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default='albert-base-v2')
    args = parser.parse_args()
    args.ncpu = mul.cpu_count()
    
    if 'albert' in args.pretrained_model:
        model_name = 'albert'
    elif 'roberta' in args.pretrained_model:
        model_name = 'roberta'
    elif 'xlnet' in args.pretrained_model:
        model_name = 'xlnet'
    elif 'bert' in args.pretrained_model:
        model_name = 'bert'

    data_collections = ['train', 'dev', 'test']
    for each in data_collections:
        args.data_dir = ROOT+'/data/raw/{}'.format(each)
        args.out_path = ROOT+'/data/pt/{}-{}.pt'.format(each, model_name)
        start = time.time()
        if not os.path.exists(args.out_path):
            preprocessor = Preprocessor(args)
            samples = preprocessor.preprocess()
            preprocessor.save(samples)
        end = time.time()
        print("time using: {}".format(end-start))
