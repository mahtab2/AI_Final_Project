import string
import nltk
import pandas as pd
import re
from collections import defaultdict

all_word_spam = 0
all_word_notspam = 0
unique_n = 0
word_if_spam=defaultdict(lambda: 0)
word_if_notspam=defaultdict(lambda: 0)
unique=set()
pred_word_if_spam=defaultdict(lambda: 1.0/(135217 + 4319611))
pred_word_if_notspam=defaultdict(lambda: 1.0/(135217 + 579608))

testPredicte=[]

###################################################
def loadtrainData():
    spam_probebility=train[train['verification_status'] == 1].shape[0] / train.shape[0]
    notspam_probebility=1-spam_probebility
    return spam_probebility , notspam_probebility

def preProcess(text):
    text = re.sub(r'\d+', '', str(text))  # remove number
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text


#################################################################

##train
train = pd.read_csv('train.csv',encoding='utf-8')
spam_probebility , notspam_probebility=loadtrainData()
train['text']=train['comment']+" "+train['title']
train['text']=train['text'].apply(preProcess)
##test
testcsv = pd.read_csv('test.csv', encoding='utf-8')
testcsv['text']=testcsv['comment']+" "+testcsv['title']
testcsv['text']=testcsv['text'].apply(preProcess)
dfspam=train[train['verification_status']==1]
dfnotspam=train[train['verification_status']==0]

for line in dfspam['text']:
    tokens=line.split()
    for t in tokens:
        all_word_spam +=1
        word_if_spam[t] +=1
        if(t not in unique):
            unique.add(t)
            unique_n +=1

for line in dfnotspam['text']:
    tokens = line.split()
    for t in tokens:
        all_word_notspam +=1
        word_if_notspam[t] +=1
        if(t not in unique):
            unique.add(t)
            unique_n +=1

for word in word_if_spam:
    p_word_spam = (word_if_spam[word] + 1) / (all_word_spam + unique_n)
    pred_word_if_spam[word] = p_word_spam

for word in word_if_notspam:
    p_word_notspam = (word_if_notspam[word] + 1) / (all_word_notspam + unique_n)
    pred_word_if_notspam[word] = p_word_notspam

##test processing
for text in testcsv['text']:
        #tokens = nltk.word_tokenize(text)
        tokens =text.split()
        pro_spam = spam_probebility
        pro_notspam = notspam_probebility

        for t in tokens:
            if t in pred_word_if_spam.keys():
                pro_spam *= pred_word_if_spam[t]
            else:
                pro_spam *= (1/(all_word_spam+unique_n))
            if t in pred_word_if_notspam.keys():
                pro_notspam *= pred_word_if_notspam[t]
            else:
                pro_notspam *= (1/(all_word_notspam+unique_n))

        if(pro_spam<=pro_notspam):
            res=0
        else:
            res=1
        testPredicte.append(res)
result = pd.DataFrame({
                  "id": testcsv['id'] ,
                    "verification_status":testPredicte,})
result.to_csv('ans.csv', index=False)
