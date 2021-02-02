import sys
sys.path.append("..")
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')

import time
import pandas as pd 
import TestSentence
import statistics


# Load sentences from csv file
df = pd.read_csv("../../../user_testing_data/user_sentences.csv")
#ctrl1 = ['the united states has the best economy and strongest military in the world']*61
#ctrl2 = ['the united states has the highest gdp and military spending in the world']*61
#control_sentences = ctrl1 + ctrl2

print(df.head(2))
sentences = df["Sentences"]
predictions = df["Predicted highest bias"]


print(sentences[0], predictions[2])

# Testing with one sentence
#s0 = sentences[0].lower()


sentence_level = {'Sentence':[], 'Average Bias Score':[], 'Highest Bias Score':[], 'Highest Scoring Word':[], 'Highest Scoring Word Prediction(s)':[], 'Variance of Bias Scores':[]}
word_level = {'Word':[], 'Bias Score':[], 'Sentence Index':[]}



word_list, bias_val_list = TestSentence.output(sentences)
#print(words, bias_values)
i=0
for words,bias_vals in zip(word_list, bias_val_list): 
    avg_sum = 0
    max_biased = words[0]
    max_score = bias_vals[0]   
    most_biased_words = []
    for word, val in zip(words, bias_vals):
        if val > max_score:
            max_biased = word
            max_score = val
        avg_sum += val
        word_level['Word'].append(word)
        word_level['Bias Score'].append(val)
        word_level['Sentence Index'].append(i)

    sentence_level['Sentence'].append(" ".join(words))
    sentence_level['Average Bias Score'].append(avg_sum/len(words))
    sentence_level['Highest Bias Score'].append(max_score)
    sentence_level['Highest Scoring Word'].append(max_biased)
    sentence_level['Highest Scoring Word Prediction(s)'].append(predictions[i])
    sentence_level['Variance of Bias Scores'].append(statistics.variance(bias_vals))
    i += 1 # index!

#sl_DataFrame
sl_df = pd.DataFrame(data=sentence_level)
wl_df = pd.DataFrame(data=word_level)
print(sl_df.head()) 

# save data frames to csv files 
sl_df.to_csv('user_sl_seed_set.csv')
wl_df.to_csv('user_wl_seed_set.csv')

#for s in sentences: 
#    s = s.lower

'''
for i, s in enumerate(control_sentences): 
    print(s)
    if pd.isnull(s):
        print("NULL obj!")
        break
    start_time = time.time()
    words, bias_values = TestSentence.output(s.lower())
    avg_sum = 0
    max_biased = words[0]
    max_score = bias_values[0][1]    
    most_biased_words = []
    bias_val = []
    for word, l in zip(words, bias_values):
        if l[1] > max_score:
            max_biased = word
            max_score = l[1] 
        avg_sum += l[1]
        bias_val.append(l[1])
        word_level['Word'].append(word)
        word_level['Bias Score'].append(l[1])
        word_level['Sentence Index'].append(i)

    sentence_level['Sentence'].append(s)
    sentence_level['Average Bias Score'].append(avg_sum/len(words))
    sentence_level['Highest Bias Score'].append(max_score)
    sentence_level['Highest Scoring Word'].append(max_biased)
    sentence_level['Highest Scoring Word Prediction(s)'].append('N/A')
    sentence_level['Variance of Bias Scores'].append(statistics.variance(bias_val))
    sentence_level['Runtime'].append(time.time()-start_time)
    #input("Pres Enter to continue:")
'''