import sys
#sys.path.append('c:\python38\lib\site-packages')
#sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
from flask import Flask, request, jsonify
import json
import TestSentence
import time
import pymongo
from pymongo import MongoClient
#from flask_pymongo import PyMongo

app = Flask(__name__)

@app.route('/api/results', methods=['POST'])
def api_post():
    print("GET RESULTS!")
    text = request.data
    texty = text.decode('utf-8')
    dictionary = json.loads(texty)
    print(dictionary['myText'])
    var = dictionary['myText'].lower()
    results = TestSentence.output(var)
    print("Adding results to DB:")
    db_res = {"sentence string": results}
    collection.insert_one(db_res)
    print("INSERTED TO DB!")
    return results #jsonify(text=results)


@app.route('/api/time')
def get_current_time():
    return {'time': time.time()}


'''@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello there!</h1>"'''

if __name__=="__main__":
    m_client = MongoClient("mongodb://3.134.119.225")
    db = m_client.sentence_results
    collection = db.res
    print("RUNNING")
    app.run(host="0.0.0.0")
