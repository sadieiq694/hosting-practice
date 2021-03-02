import sys
sys.path.append("..")
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')

import time
import pandas as pd 
import TestSentence
import statistics



article = 'America has the best economy and strongest military in the world America has the best economy and strongest military in the world. our recent elections were rigged and fraudulent'
sentences = article.split('.')
print(sentences)
input()
sentences_l = [s.lower() for s in sentences]
print(sentences_l)
word_list, bias_list = TestSentence.output(sentences_l)
print(word_list, bias_list)
print(bias_list[0]==bias_list[1])