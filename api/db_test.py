import sys
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
import pymongo
from pymongo import MongoClient

# pymongo test

client = MongoClient("mongodb://3.134.119.225")

db = client.sentence_results

document = {"TEST ENTRY": "hi my name is Sadie"}
#db.res.insert_one(document)

print(db.res.count())

print(db.res.find_one(sort=[( '_id', pymongo.DESCENDING )]))

