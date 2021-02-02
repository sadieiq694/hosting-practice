import sys
#sys.path.append('c:\python38\lib\site-packages')
#sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
from flask import Flask, request, jsonify
import json
import TestSentence
import time
import string
import pymongo
from pymongo import MongoClient
#from flask_pymongo import PyMongo

app = Flask(__name__)

m_client = MongoClient("mongodb://3.134.119.225")
db = m_client.sentence_results
collection = db.res

@app.route('/api/results', methods=['POST'])
def api_post():
    print("GET RESULTS!")
    start_time = time.time()

    # get text and format it
    text = request.data
    texty = text.decode('utf-8')
    #texty.translate(str.maketrans('', '', string.punctuation))
    dictionary = json.loads(texty) # why are we loading it into a dictionary and then back out? 
    print(dictionary['myText'])
    var = dictionary['myText'].lower()
    # remove punctuation here 

    # Run through algorithm 
    words, bias_values =  TestSentence.output(var)

    results = {} # will have "runtime", "sentence_results"
    results['sentence_resluts'] = []

    for w, b in zip(words, bias_values):

        # Format output string 
        # starts as python dictionary which we will convert to a json string
        outWordsScores = []
        avg_sum = 0
        max_biased = words[0]
        max_score = bias_values[0][1]    
        most_biased_words = []
        output = ""
        for word, score in zip(w, b):
            if score > max_score:
                max_biased = word
                max_score = score
            avg_sum += score
            outWordsScores.append(word + ": " + "{:.5f}".format(score) + " ")
            if score >= 0.45:
                most_biased_words.append(word)

        s_level_results = {
            "words" : outWordsScores,
            "average": "{:.5f}".format(avg_sum/len(words)),
            "max_biased_word": max_biased.upper() + ": " + "{:.5f}".format(max_score),
            "max_word": max_biased.upper(),
        } 

        results['sentence_resluts'].append(s_level_results)

        print("Average bias: ", avg_sum/len(words))

        # Insert data to database # change this to match test article analysis
        db_entry = {}
        db_entry['words'] = words
        db_entry['bias_vals'] = bias_values
        db_entry['most_biased'] = most_biased_words
        print("Adding results to DB:")
        collection.insert_one(db_entry)
        print("INSERTED TO DB!")

    results['runtime'] = str(time.time() - start_time) + " seconds\n"

    '''
    results = {
        "runtime": 4,
        "sentence_results":
        [
            {
                "words":[w1,w2],
                "average":0.2,
                ....
            },
            ...
            {}
        ]
    } 
    '''

    return json.dumps(results) #jsonify(text=results)


@app.route('/api/time')
def get_current_time():
    return {'time': time.time()}


'''@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello there!</h1>"'''

if __name__=="__main__":
    print("RUNNING")
    app.run(host="0.0.0.0")
