from flask import Flask, request, jsonify
import json
import TestSentence

app = Flask(__name__)

@app.route('/api/result', methods=['POST'])
def api_post():
    text = request.data
    texty = text.decode('utf-8')
    dictionary = json.loads(texty)
    print(dictionary['myText'])
    var = dictionary['myText'].lower()
    results = TestSentence.output(var)
    return results #jsonify(text=results)


@app.route('/api/time')
def get_current_time():
    return {'time': time.time()}


'''@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello there!</h1>"'''

if __name__=="__main__":
    app.run(host="0.0.0.0")
