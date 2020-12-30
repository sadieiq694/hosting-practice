import sys
import time
#sys.path.append('c:\python38\lib\site-packages')
#sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
import pytorch_pretrained_bert.modeling as modeling
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import nltk
from nltk.parse.stanford import StanfordDependencyParser

from tqdm import tqdm
import sys

import pickle
import os
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
#import argparse
import sklearn.metrics as metrics
#from simplediff import diff
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertModel, BertSelfAttention
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from features import FeatureGenerator

CUDA = (torch.cuda.device_count() > 0)
if CUDA:
    print("GPUS!")
    input()

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


class AddCombine(nn.Module):
    def __init__(self, hidden_dim, feat_dim, layers, dropout_prob, small=False,
            out_dim=-1, pre_enrich=False, include_categories=False,
            category_emb=False, add_category_emb=False):
        super(AddCombine, self).__init__()

        self.include_categories = include_categories
        if include_categories:
            feat_dim += 43

        if layers == 1:
            self.expand = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.Dropout(dropout_prob))
        else:
            waist_size = min(feat_dim, hidden_dim) if small else max(feat_dim, hidden_dim)
            self.expand = nn.Sequential(
                nn.Linear(feat_dim, waist_size),
                nn.Dropout(dropout_prob),
                nn.Linear(waist_size, hidden_dim),
                nn.Dropout(dropout_prob))
        
        if out_dim > 0:
            self.out = nn.Linear(hidden_dim, out_dim)
        else:
            self.out = None

        if pre_enrich:
            self.enricher = nn.Linear(feature_size, feature_size)        
        else:
            self.enricher = None

        # manually set cuda because module doesn't see these combiners for bottom         
        if CUDA:
            self.expand = self.expand.cuda()
            if out_dim > 0:
                self.out = self.out.cuda()
            if self.enricher is not None:
                self.enricher = self.enricher.cuda()

    def forward(self, hidden, feat, categories=None):
        if self.include_categories:
            categories = categories.unsqueeze(1)
            categories = categories.repeat(1, features.shape[1], 1)
            if self.add_category_emb:
                features = features + categories
            else:
                features = torch.cat((features, categories), -1)

        if self.enricher is not None:
            feat = self.enricher(feat)
    
        combined = self.expand(feat) + hidden
    
        if self.out is not None:
            return self.out(combined)

        return combined

class BertForMultitaskWithFeatures(BertPreTrainedModel): 
    
    def __init__(self, config, lex_path, cls_num_labels=2, tok_num_labels=2, tok2id=None, lexicon_feature_bits=1):
        super(BertForMultitaskWithFeatures, self).__init__(config)

        self.bert = BertModel(config)

        self.featureGenerator = FeatureGenerator(
            POS2ID, REL2ID, tok2id=tok2id, lexicon_path=lex_path, pad_id=0, 
            lexicon_feature_bits=lexicon_feature_bits)
        #print("Printing pad id:", self.featureGenerator.pad_id)
        nfeats = 90 if lexicon_feature_bits == 1 else 118; 

        # hidden_size = 512
        # nfeats
        # combiner_layers = 1
        # hidden_dropout_prob = 
        # small_waist = false
        # out dim
        # pre_enrich = false
        # include_categories = False
        # category_emb = 
        # add_category_emb = 
        self.tok_classifier = AddCombine(config.hidden_size, nfeats, 1,
                config.hidden_dropout_prob, False,
                out_dim=tok_num_labels, pre_enrich=False,
                include_categories=False,
                category_emb=False,
                add_category_emb=False)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                rel_ids=None, pos_ids=None, categories=None, pre_len=None):
        
        #featurize_batch(self, batch_ids, rel_ids, pos_ids, padded_len=0)
        features = self.featureGenerator.featurize_batch(
            input_ids.detach().cpu().numpy(), 
            rel_ids, #.detach().cpu().numpy(), 
            pos_ids, #.detach().cpu().numpy(), 
            padded_len=input_ids.shape[1])
        features = torch.tensor(features, dtype=torch.float)
        if CUDA:
            features = features.cuda()

        # make sure all inputs on CPU
        #print("TYPES:", type(input_ids), type(token_type_ids), type(attention_mask))
        #sequence_output, pooled_output = self.bert(
        #    input_ids.detach().cuda(), token_type_ids, attention_mask.detach().cpu(), output_all_encoded_layers=False)
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.cls_dropout(pooled_output)
        cls_logits = self.cls_classifier(pooled_output)

        '''if ARGS.category_emb:
            categories = self.category_embeddings(
                categories.max(-1)[1].type(
                    'torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))'''

        tok_logits = self.tok_classifier(sequence_output, features, categories)

        return cls_logits, tok_logits

class BertForMultitask(BertPreTrainedModel):

    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(BertForMultitask, self).__init__(config)
        self.bert = BertModel(config)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)
        
        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)
        
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        labels=None, rel_ids=None, pos_ids=None, categories=None, pre_len=None):
        #global ARGS
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        cls_logits = self.cls_classifier(pooled_output)
        cls_logits = self.cls_dropout(cls_logits)

        # NOTE -- dropout is after proj, which is non-standard
        tok_logits = self.tok_classifier(sequence_output)
        tok_logits = self.tok_dropout(tok_logits)

        return cls_logits, tok_logits