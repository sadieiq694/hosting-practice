# Imports
#from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertSelfAttention
import sys
import time
#sys.path.append('c:\python38\lib\site-packages')
#sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
#sys.path.append('..\..\ML')

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_pretrained_bert.modeling as modeling
#import copy
#from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tqdm import tqdm
import sys

import pickle
import os
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
#from tensorboardX import SummaryWriter
#import argparse
import sklearn.metrics as metrics
#from simplediff import diff
from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertModel, BertSelfAttention
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from features import FeatureGenerator # might not need this
from models import AddCombine, BertForMultitaskWithFeatures, BertForMultitask

# Set the seed value all over the place to make this reproducible.
# This should get rid of non-determinism

CUDA = (torch.cuda.device_count() > 0)
if CUDA:
    print("GPUS!")
    input()

#####################
### DIF #############
#####################
def softmax(x, axis=None):
  x=x-x.max(axis=axis, keepdims=True)
  y= np.exp(x)
  return y/y.sum(axis=axis, keepdims=True)

def diff(old, new):

    # Create a map from old values to their indices
    old_index_map = dict()
    for i, val in enumerate(old):
        old_index_map.setdefault(val,list()).append(i)


    overlap = dict()

    sub_start_old = 0
    sub_start_new = 0
    sub_length = 0

    for inew, val in enumerate(new):
        _overlap = dict()
        for iold in old_index_map.get(val,list()):
            # now we are considering all values of iold such that
            # `old[iold] == new[inew]`.
            _overlap[iold] = (iold and overlap.get(iold - 1, 0)) + 1
            if(_overlap[iold] > sub_length):
                # this is the largest substring seen so far, so store its
                # indices
                sub_length = _overlap[iold]
                sub_start_old = iold - sub_length + 1
                sub_start_new = inew - sub_length + 1
        overlap = _overlap

    if sub_length == 0:
        # If no common substring is found, we return an insert and delete...
        return (old and [('-', old)] or []) + (new and [('+', new)] or [])
    else:
        # ...otherwise, the common substring is unchanged and we recursively
        # diff the text before and after that substring
        return diff(old[ : sub_start_old], new[ : sub_start_new]) + \
               [('=', new[sub_start_new : sub_start_new + sub_length])] + \
               diff(old[sub_start_old + sub_length : ],
                       new[sub_start_new + sub_length : ])

##########################################################
#### SET UP ID DICTIONARIES FOR WORDS, RELATIONS, POS ####
##########################################################
## UPDATE THESE!!!
DATA_DIRECTORY = 'data/'
LEXICON_DIRECTORY = DATA_DIRECTORY + 'lexicons/'
PRYZANT_DATA = DATA_DIRECTORY + 'bias_data/WNC/'
#IMPORTS = 
training_data = PRYZANT_DATA + 'biased.word.train'
testing_data = PRYZANT_DATA + 'biased.word.test'
#categories_file = PRYZANT_DATA + 'revision_topics.csv'
pickle_directory = 'pickle_dir/'
cache_dir = DATA_DIRECTORY + 'cache/'
model_save_dir = 'saved_models/'


RELATIONS = [
  'det', # determiner (the, a)
  'amod', # adjectival modifier
  'nsubj', # nominal subject
  'prep', # prepositional modifier
  'pobj', # object of preposition
  'ROOT', # root
  'attr', # attribute
  'punct', # punctuation
  'advmod', # adverbial modifier
  'compound', # compound
  'acl', # clausal modifier of noun (adjectivial clause)
  'agent', # agent
  'aux', # auxiliary
  'ccomp', # clausal complement
  'dobj', # direct object
  'cc', # coordinating conjunction 
  'conj', # conjunct
  'appos', # appositional 
  'nsubjpass', # nsubjpass
  'auxpass', # auxiliary (passive)
  'poss', # poss
  'nummod', # numeric modifier
  'nmod', # nominal modifier
  'relcl', # relative clause modifier
  'mark', # marker
  'advcl', # adverbial clause modifier
  'pcomp', # complement of preposition
  'npadvmod', # noun phrase as adverbial modifier
  'preconj', # pre-correlative conjunction
  'neg', # negation modifier
  'xcomp', # open clausal complement
  'csubj', # clausal subject
  'prt', # particle
  'parataxis', # parataxis
  'expl', # expletive
  'case', # case marking
  'acomp', # adjectival complement
  'predet', # ??? 
  'quantmod', # modifier of quantifier
  'dep', # unspecified dependency
  'oprd', # object predicate
  'intj', # interjection
  'dative', # dative
  'meta', # meta modifier
  'csubjpass', # clausal subject (passive)
  '<UNK>' # unknown
]

REL2ID = {x: i for i,x in enumerate(RELATIONS)}

# PARTS OF SPEECH
POS_TAGS = [
  'DET', # determiner (a, an, the)
  'ADJ', # adjective (big, old, green, first)
  'NOUN', # noun (girl, cat, tree)
  'ADP', # adposition (in, to, during)
  'NUM', # numeral (1, 2017, one, IV)
  'VERB', # verb (runs, running, eat, ate)
  'PUNCT', # punctuation (., (, ), ?)
  'ADV', # adverb (very, tomorrow, down)
  'PART', # particle ('s, not)
  'CCONJ', # coordinating conjunction (and, or, but)
  'PRON', # pronoun(I, you, he, she)
  'X', # other (fhefkoskjsdods)
  'INTJ', # interjection (hello, psst, ouch, bravo)
  'PROPN', # proper noun (Mary, John, London, HBO) 
  'SYM', # symbol ($, %, +, -, =)
  '<UNK>' # unknown
]

POS2ID = {x: i for i, x in enumerate(POS_TAGS)}

EDIT_TYPE2ID = {'0':0, '1':1, 'mask':2}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', os.getcwd() + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

# BERT initialization params
config = 'bert-base-uncased'
cls_num_labels = 43
tok_num_labels = 3
tok2id = tok2id


# PADDING TO MAX_SEQ_LENGTH
def pad(id_arr, pad_idx):
  max_seq_len = 60
  return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

def to_probs(logits, lens):
    #print(logits)
    per_tok_probs = softmax(np.array(logits)[:, :, :2], axis=2)
    pos_scores = per_tok_probs[-1, :, :]
    out = []
    #for score_seq, l in zip(pos_scores, lens):
    out.append(pos_scores[:].tolist())
    return out

# Take one sentence ... 
def run_inference(model, ids, tokenizer):

    #global ARGS
    # we will pass in one sentence, no post_toks, 

    out = {
        'input_toks': [],
        #'tok_loss': [],
        'tok_logits': [],
        'tok_probs': []
        #'tok_labels': [],
        #'labeling_hits': []
        #'input_len': 0
    }

    #for step, batch in enumerate(tqdm(eval_dataloader)):
        #if False and step > 2:
        #    continue
    #if CUDA:
    #    batch = tuple(x.cuda() for x in batch)
    pre_len = len(ids)

    with torch.no_grad():
        _, tok_logits = model(ids, attention_mask=None,
            rel_ids=None, pos_ids=None, categories=None,
            pre_len=None) # maybe pre_len
        #tok_loss = loss_fn(tok_logits, tok_label_id, apply_mask=tok_label_id)
    out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in ids.cpu().numpy()]
    #out['post_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in post_in_id.cpu().numpy()]
    #out['tok_loss'].append(float(tok_loss.cpu().numpy()))
    logits = tok_logits.detach().cpu().numpy()
    #labels = tok_label_id.cpu().numpy()
    out['tok_logits'] += logits.tolist()
    #out['tok_labels'] += labels.tolist()
    out['tok_probs'] += to_probs(logits, pre_len)
    #out['input_len'] = pre_len
    #out['labeling_hits'] += tag_hits(logits, labels)

    return out


def test_sentence(model, s): 
    tokens = tokenizer.tokenize(s)
    #print(tokens)
    length = len(tokens)
    ids = pad([tok2id.get(x, 0) for x in tokens], 0)

    #print(ids)
    ids = torch.LongTensor(ids)
    #print(ids, ids.size())
    ids = ids.unsqueeze(0)
    #print(ids, ids.size(1))
    
    model.eval()
    output = run_inference(model, ids, tokenizer)
    return output, length

def output(sentences):
    # Takes a list of sentences = [s1, s2, s3]
    # Returns list of tokens and list of corresponding bias scores for each sentence
    #   So, list of lists: 
    #       word_list = [[w1,w2,...wn_1], . . . [w1, w2, ... wn_2]]
    #       bias_list = [[0.1,0.33, ... 0.02], . . .[0.9, 0.002, ... 0.5]]

    # using new models with linguistic features
    model = BertForMultitaskWithFeatures.from_pretrained(
        'bert-base-uncased', LEXICON_DIRECTORY,
        cls_num_labels=cls_num_labels,
        tok_num_labels=tok_num_labels,
        tok2id=tok2id, 
        lexicon_feature_bits=1)

    # Load model
    #print("Loading Model")
    saved_model_path = model_save_dir + 'features.ckpt'
    model.load_state_dict(torch.load(saved_model_path, map_location=torch.device("cpu")))

    word_list = []
    bias_list = []
    for sentence in sentences: 
        print(sentence)
        out, length = test_sentence(model, sentence) 
        #print("Results:")

        bias_val = out['tok_probs'][0][:length]
        prob_bias = [b[1] for b in bias_val]

        word_list.append(out['input_toks'][0][:length])
        bias_list.append(prob_bias)

    return word_list, bias_list 