import sys
#sys.path.append("c:\python38\lib\site-packages")

#from newsapi import NewsApiClient
from newsapi.newsapi_client import NewsApiClient
import json
import random
import requests

# Init
api_key_2 = 'b6a1f64d144b43c1bba5370b62d879e0'
# Old one:  c1ff125522de4c749e615dca5cba6eb5
newsapi = NewsApiClient(api_key='b6a1f64d144b43c1bba5370b62d879e0')
    
def getarticles(inp):
    
    #centerist source call
    url1 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        'from=2021-02-09&'
        'sortBy=relevancy&'
        'sources=associated-press&'
        'domains= reuters.com,cbsnews.com,abcnews.go.com,bloomburg.com,economist.com,forbes.com,cnbc.com,thehill.com,politico.com$'
        'page = 1&'
        'apiKey=b6a1f64d144b43c1bba5370b62d879e0')

    response1 = requests.get(url1)
    articles1 = response1.json()

    #skew left source call
    url2 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        'from=2021-02-09&'
        'sortBy=relevancy&'
        'sources=cnn&'
        'domains= nytimes.com,theguardian.com,msnbc.com,theatlantic.com,vox.com,washingtonpost.com,huffpost.com,thedailybeast.com&'
        'pageSize=100&'
        'page = 1&'
        'apiKey=b6a1f64d144b43c1bba5370b62d879e0')

    response2 = requests.get(url2)
    articles2 = response2.json()

    #partisan left source call
    url3 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        'from=2021-02-09&'
        'sortBy=relevancy&'
        'domains= slate.com,jacobinmag.com,rawstory.com,progressive.org&'
        'pageSize=100&'
        'page = 1&'
        'apiKey=b6a1f64d144b43c1bba5370b62d879e0')

    response3 = requests.get(url3)
    articles3 = response3.json()

    #partisan right source call
    url4 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        'from=2021-02-09&'
        'sortBy=relevancy&'
        'sources=fox-news&'
        'domains= dailywire.com,dailycaller.com,nationalreview.com&'
        'pageSize=100&'
        'page = 1&'
        'apiKey=b6a1f64d144b43c1bba5370b62d879e0')

    response4 = requests.get(url4)
    articles4 = response4.json()

    #skew right source call
    url5 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        'from=2021-02-09&'
        'sortBy=relevancy&'
        'sources=the-wall-street-journal&'
        'domains= reason.com,quillette.com,realclearpolitics.com,nypost.com,washingtonexaminer.com,rasmussenreports.com,freebeacon.com,zerohedge.com&'
        'pageSize=100&'
        'page = 1&'
        'apiKey=b6a1f64d144b43c1bba5370b62d879e0')

    response5 = requests.get(url5)
    articles5 = response5.json()

    #politcal catagory arrays
    middle = ['']
    sleft = ['']
    sright = ['']
    pleft = ['']
    pright = ['']

    #all urls being sorted from json files into political catagory arrays
    for x in articles1['articles']:
        middle.append(x['url'])

    for x in articles2['articles']:
        sleft.append(x['url'])

    for x in articles3['articles']:
        pleft.append(x['url'])

    for x in articles4['articles']:
        pright.append(x['url'])

    for x in articles5['articles']:
        sright.append(x['url'])

    # print("PARTISAN LEFT: ", pleft)
    # print("SKEWS LEFT: ", sleft)
    # print("MIDDLE: ", middle)
    # print("SKEWS RIGHT: ", sright)
    # print("PARTISAN RIGHT: ", pright)

    #this removes the blank space placeholder for each array
    if len(pleft) > 1:
        pleft.remove('')
    if len(sleft) > 1:
        sleft.remove('')
    if len(middle) > 1:
        middle.remove('')
    if len(sright) > 1:
        sright.remove('')
    if len(pright) > 1:
        pright.remove('')

    #random selection for each political catagory
    related = {}
    related['partisan_left'] = random.choice(pleft)
    related['skews_left'] = random.choice(sleft)
    related['middle'] = random.choice(middle)
    related['skews_right'] = random.choice(sright)
    related['partisan_right'] = random.choice(pright)

    print(related)

    #returns array of articles, elements 0-4 is partisan left to partisan right respectively
    return related

# print("STARTING")
# getarticles('trump impeachment')