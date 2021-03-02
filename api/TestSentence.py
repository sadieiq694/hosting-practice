#################
#### Imports ####
#################
import sys
import time
import os

sys.path.append('../../ML')
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
sys.path.append('..\..\ML')

import numpy as np
import statistics

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# pretrained BERT imports
import pytorch_pretrained_bert.modeling as modeling
from pytorch_pretrained_bert.modeling import BertModel, BertSelfAttention, BertPreTrainedModel, BertForTokenClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

# other user scripts
from myfeatures import FeatureGenerator # might not need this
from models import AddCombine, BertForMultitaskWithFeatures #, BertForMultitask

CUDA = (torch.cuda.device_count() > 0)
if CUDA:
    print("GPUS!")
    input()

#####################
### Softmax #############
#####################
def softmax(x, axis=None):
  x=x-x.max(axis=axis, keepdims=True)
  y= np.exp(x)
  return y/y.sum(axis=axis, keepdims=True)

##########################################################
#### SET UP ID DICTIONARIES FOR WORDS, RELATIONS, POS ####
##########################################################
## UPDATE THESE!!!
DATA_DIRECTORY = 'data/'
LEXICON_DIRECTORY = DATA_DIRECTORY + 'lexicons/'
cache_dir = DATA_DIRECTORY + 'cache/'
model_save_dir = 'saved_models/'
saved_model_path = 'saved_models/features.ckpt'

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

# BERT initialization params
config = 'bert-base-uncased'
cls_num_labels = 43
tok_num_labels = 3

tokenizer = BertTokenizer.from_pretrained(config, os.getcwd() + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


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
def run_inference(model, ids): #, tokenizer):
    #global ARGS
    # we will pass in one sentence, no post_toks, 

    out = {
        'input_toks': [], # text input
        'tok_logits': [],
        'tok_probs': [] # bias probabilities
    }

    pre_len = len(ids)

    with torch.no_grad():
        _, tok_logits = model(ids, attention_mask=None,
            rel_ids=None, pos_ids=None, categories=None,
            pre_len=None) # maybe pre_len
    
    out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in ids.cpu().numpy()]
    logits = tok_logits.detach().cpu().numpy()
    out['tok_logits'] += logits.tolist()
    out['tok_probs'] += to_probs(logits, pre_len)

    return out

def test_sentence(model, s): 
    tokens = tokenizer.tokenize(s)
    length = len(tokens)
    
    # get tokens from BERT
    ids = pad([tok2id.get(x, 0) for x in tokens], 0)
    ids = torch.LongTensor(ids)
    ids = ids.unsqueeze(0)
    
    model.eval() # constant random seed
    output = run_inference(model, ids) #, tokenizer)
    return output, length

def changeRange(old_range, new_range, value):
    # given an old range, new range, and value in the old range, 
    # maps it to the new range
    # we will use old_range[0,1] new_range [0,10]
    (old_min, old_max), (new_min, new_max) = old_range, new_range
    return  new_min + ((value - old_min) * (new_max - new_min) / (old_max - old_min))

def output(sentences):
    results = {}
    word_score_list = []
    preformatted_words = []
    preformatted_scores = []
    results['sentence_results'] = []
    #print('sentences:', sentences)
    #print("New testsentence code!")
    # Takes a list of sentences = [s1, s2, s3]
    # Returns list of tokens and list of corresponding bias scores for each sentence
    #   So, list of lists: 
    #       word_list = [[w1,w2,...wn_1], . . . [w1, w2, ... wn_2]]
    #       bias_list = [[0.1,0.33, ... 0.02], . . .[0.9, 0.002, ... 0.5]]

    # using new models with linguistic features
    model = BertForMultitaskWithFeatures.from_pretrained(
        config, LEXICON_DIRECTORY,
        cls_num_labels=cls_num_labels,
        tok_num_labels=tok_num_labels,
        tok2id=tok2id, 
        lexicon_feature_bits=1)

    # Load model
    saved_model_path = model_save_dir + 'features.ckpt'
    model.load_state_dict(torch.load(saved_model_path, map_location=torch.device("cpu")))

    word_list = []
    bias_list = []
    for sentence in sentences:
        sentence=sentence.lower() 
        #print(sentence)
        out, length = test_sentence(model, sentence) 
        #print("Results:")

        bias_val = out['tok_probs'][0][:length]
        prob_bias = [b[1] for b in bias_val]

        word_list.append(out['input_toks'][0][:length])
        bias_list.append(prob_bias)

    #print("LENGTHS:", len(word_list), len(bias_list))

    scaled_bias_scores = []
    num = 0
    for words, biases in zip(word_list, bias_list):
        # Format output string 
        # starts as python dictionary which we will convert to a json string
        outWordsScores = []
        avg_sum = 0
        max_biased = words[0]
        max_score = biases[0]   
        most_biased_words = []
        for word, score in zip(words, biases):
            preformatted_words.append(word)
            preformatted_scores.append(score)
            if score > max_score:
                max_biased = word
                max_score = score
            avg_sum += score
            if len(word) >= 3 and word[:2] == "##":
                # stuff
                last_word_score = outWordsScores[-1]
                #print(last_word_score, word, score)
                outWordsScores[-1][0] = last_word_score[0] + word[2:]
                outWordsScores[-1][1] = max(last_word_score[1], score)
            else:
                outWordsScores.append([word, score])
            if score >= 0.45:
                most_biased_words.append(word)
        
        # one of these per sentence
        bias_score = changeRange([0,1], [0,10], max_score)
        scaled_bias_scores.append(bias_score)
        #print("Scaled bias scores: ", scaled_bias_scores)

        #print("max biased and max score:", max_biased, max_score)
        num = num + 1
        s_level_results = {
            "words" : outWordsScores,
            "average": "{:.5f}".format(avg_sum/len(words)),
            "max_biased_word": max_biased + ": " + "{:.5f}".format(max_score),
            "bias_score":bias_score,
            "order":num
        } 

        results['sentence_results'].append(s_level_results)
    
    '''formatted_words = []
    formatted_scores = []
    for word, score in zip(preformatted_words, preformatted_scores):
        if len(word) >= 3 and word[:2] == "##":
            # stuff
            last_word = formatted_words[-1]
            formatted_words[-1] = last_word + word[2:]
            last_score = formatted_scores[-1]
            formatted_scores[-1] = (last_score + score)/2
        else:
            formatted_words.append(word)
            formatted_scores.append(score)

    print(len(formatted_scores), len(formatted_words))
    for word, score in zip(formatted_words, formatted_scores): 
        word_score_list.append({'word':word, 'score':score}) # add type later!
    results['article_score'] = statistics.mean(top_twenty_fifth)'''


    # Full article data
    # Sort scaled bias score largest to smallest: 
    scaled_bias_scores.sort(reverse=True)
    upper_bound = int(len(scaled_bias_scores)/2)

    if upper_bound == 0:
        upper_bound = 1

    top_twenty_fifth = scaled_bias_scores[:upper_bound]
    results['word_list'] = word_score_list
    results['article_score'] = statistics.mean(top_twenty_fifth)


    #print(results['article_score'])

    #print('DONE IN TEST SENTENCE')
    
    return results 