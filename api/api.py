import sys
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
sys.path.append('..\\..\\Related_Articles')
sys.path.append('../../Related_Articles')
from flask import Flask, request, jsonify
import json
import TestSentence
import time
import string
import newscraper
#import pymongo
#from pymongo import MongoClient
import nltk.data
import RelatedArticles_five_calls #import getarticles

app = Flask(__name__)

'''m_client = MongoClient("mongodb://3.134.119.225")
db = m_client.sentence_results
collection = db.res'''

def analyze_sentences(text, start_time):
    # Split into multiple sentences here
    nltk.download('punkt')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)
    #sentences = var.split('. ')
    #print(sentences)

    # Run through algorithm 
    results = TestSentence.output(sentences)
    
    # Insert data to database # change this to match test article analysis
    '''db_entry = {}
    db_entry['words'] = words
    db_entry['bias_vals'] = bias_values
    db_entry['most_biased'] = most_biased_words
    print("Adding results to DB:")
    collection.insert_one(db_entry)
    print("INSERTED TO DB!")'''

    #print('SENTENCE RESULTS!', results['sentence_results'])
    results['runtime'] = str(time.time() - start_time) + " seconds\n"
    #return_res = results['sentence_results']

    # REMERGE TOKENIZED WORDS (BID ##EN = BIDEN)
    # Make sure sentence parsing is working?!?!?
    return results

@app.route('/result', methods=['POST'])
def api_post():
    print("GET RESULTS!")
    start_time = time.time() # to keep track of analysis runtime

    # get text and format it
    text = request.data
    #print("raw text:", text)
    texty = text.decode('utf-8')
    #texty.translate(str.maketrans('', '', string.punctuation))
    dictionary = json.loads(texty) # why are we loading it into a dictionary and then back out? 
    #print(dictionary['myText'])
    var = dictionary['myText'] #.lower()

    results = analyze_sentences(var, start_time)

    return results

@app.route('/scrape', methods=['POST'])
def scrape_article():
    start_time = time.time() # to keep track of analysis runtime
    url = request.data
    url = url.decode('utf-8')
    url = json.loads(url)
    print(url)
    url = url['input_url']
    print("PARSING URL")
    res = newscraper.article_parse(url)
    #print(res)
    res_obj = json.loads(res) 
    #print(res_obj)
    # res.title, res.author, res.feedText, res.date, res.meta (?)
    results = analyze_sentences(res_obj['feedText'], start_time)
    res_obj['bias_results'] = results

    ####
    # function gets keywords from title, looks for frequent words in article itself??
    separate_words = res_obj['title'].split(' ')
    ####
    #print(separate_words) # remove 'the', 'a', etc in the future
    filtered_separate_words = [i for i in separate_words if i != "the" and i != "a" and i != "in" and i != "to" and i != "his" and i != "her" and i != "and" and i != "on" and i != "with"]
    related_articles = RelatedArticles_five_calls.getarticles(" ".join(filtered_separate_words[:3]))
    # Call function # return dictionary of {"left":url1, "left-leaning":url2 etc.}
    res_obj['related'] = related_articles
    ####
    return res_obj

